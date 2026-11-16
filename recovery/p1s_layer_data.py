"""Synthetic training data for spotting print failures layer by layer, seen by the P1S chamber camera.

Each sample is one moment of a print:
1. a random part (parts/*.yaml, parts/print/*/*.yaml, or one of the random training shapes in
   recovery/p1s_shapes.py, mostly round) is printed up to layer k. With probability
   p_fail, a failure from recovery/deform.py started a few layers earlier: collapse (a wall sags in),
   warp (a wall curls out), shift (the layers above slide sideways), under_extrusion (a wall section is
   thin or open) or detach (the part came off the bed and only a stub keeps printing);
2. the camera frame after layer k-1 and the frame after layer k are rendered (recovery/p1s_view.py).
   The bed is one layer lower in the second, exactly as on the printer;
3. the PLAN up to layer k is rendered from the same camera without the scene: that silhouette is the
   guide a deployed model also has (from the sliced G-code and the calibration);
4. the mask marks where the print differs from the plan (depth of print vs plan differs > 0.1 mm).

    python -m recovery.p1s_layer_data --n 5000 --out data/p1s_layer --workers 12
"""
from . import p1s_view as V                 # first: selects the EGL backend before MuJoCo loads

import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .deform import random_failure
from .p1s_dent_data import _round_and_wobble, base_camera, clean_mask, sample_look, sample_placement, _git_commit
from .p1s_shapes import shape_part
from .parts import get_part

FAILURES = ("collapse", "warp", "shift", "under_extrusion", "detach")
KINDS = ("ok",) + FAILURES
CHANGE_MM = 0.1


def default_parts():
    return sorted(glob.glob("parts/*.yaml") + glob.glob("parts/print/*/*.yaml"))


# failure types the detector found hardest (sagging and thin walls) are drawn more often
KIND_WEIGHTS = dict(collapse=1.3, warp=1.0, shift=0.8, under_extrusion=1.6, detach=0.6)


def default_config(**kw):
    """shapes: how many random training shapes (recovery/p1s_shapes.py) join the real parts, drawn with
    probability p_shape. kind_weights: how often each failure type is drawn (None = equally)."""
    cfg = dict(parts=default_parts(), shapes=160, p_shape=0.6, kind_weights=KIND_WEIGHTS, p_fail=0.6, since_layers=(1, 12), bed_range_mm=(40.0, 150.0), part_xy=None,
               yaw_sigma_deg=10.0, yaw_flip_p=0.15, camera_jitter=True, camera_error=1.0, calib=None, seed=0,
               val_every=10)
    cfg.update(kw)
    return cfg


_CACHE = {}
CACHE_PARTS = 16            # parts kept per worker (~15 MB each); with 160 shapes an unbounded cache ran out of RAM

# A worker also keeps ~4 MB per sample it renders that is never freed, so pool workers are replaced after
# this many samples, and the default worker count is what the free RAM allows.
TASKS_PER_WORKER = 50
WORKER_GB = 1.5


def safe_workers(requested=None, log=print):
    """Workers that fit in the available RAM (keeping 3 GB free for the rest of the machine)."""
    try:
        avail_gb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20
    except (OSError, StopIteration):
        avail_gb = 8.0
    fit = max(1, int((avail_gb - 3.0) / WORKER_GB))
    cpu = max(1, min(12, (mp.cpu_count() or 2) - 2))
    if requested is None:
        return min(fit, cpu)
    if requested > fit:
        log(f"--workers {requested} needs ~{requested * WORKER_GB:.0f} GB but {avail_gb:.1f} GB is free: using {fit}")
        return fit
    return requested


def _nominal(path):
    """(Part, nominal layers) for a part YAML or a training shape `shape:<n>`."""
    if path in _CACHE:
        _CACHE[path] = _CACHE.pop(path)        # most recently used last
    else:
        while len(_CACHE) >= CACHE_PARTS:
            _CACHE.pop(next(iter(_CACHE)))
        part = shape_part(int(path[6:])) if path.startswith("shape:") else get_part(path)
        _CACHE[path] = (part, part.layers())
    return _CACHE[path]


def stack_bounds(layers):
    b = np.array([l.region.bounds for l in layers if not l.region.is_empty])
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())


def printed_stack(nominal, failure, k):
    """Layers 0..k-1 as printed: the failure deforms every layer it is active on (it persists)."""
    out = []
    for i in range(k):
        lay = nominal[i]
        out.append(failure.apply(lay, i) if failure is not None and failure.active(i) else lay)
    return out


def _split(a, b):
    """First index where two stacks differ (shared layers can use one mesh)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x is not y:
            return i
    return min(len(a), len(b))


def plan_silhouette(cam, look, plan, xy, yaw, drop=0.0):
    """Where the plan's layers so far appear in the image (uint8 0/255): the guide channel."""
    with V.P1SScene(cam, look, {"plan": V.stack_mesh(plan, bulge_mm=0.0)}, xy, yaw, plan[-1].z, drop) as s:
        _, d = s.render(["plan"], scene=False)
    return (d < 0.95 * d.max()).astype(np.uint8) * 255


def pick_part(rng, cfg):
    path = cfg["parts"][rng.integers(len(cfg["parts"]))]
    if cfg.get("shapes") and rng.random() < cfg.get("p_shape", 0.5):
        path = f"shape:{rng.integers(cfg['shapes'])}"
    return path


def pick_kind(rng, cfg):
    w = cfg.get("kind_weights")
    if not w:
        return FAILURES[rng.integers(len(FAILURES))]
    p = np.array([w[k] for k in FAILURES], float)
    return FAILURES[rng.choice(len(FAILURES), p=p / p.sum())]


def make_sample(i, cfg):
    """Returns (prev_rgb, cur_rgb, guide, mask, label, (photo_prev, photo_cur))."""
    rng = np.random.default_rng([cfg["seed"], i])
    path = pick_part(rng, cfg)
    part, nominal0 = _nominal(path)
    n = len(nominal0)
    nominal = _round_and_wobble(nominal0, rng, float(rng.uniform(0.0, 0.3)), float(rng.uniform(0.0, 0.03)))
    failure, kind = None, "ok"
    if rng.random() < cfg["p_fail"]:
        kind = pick_kind(rng, cfg)
        failure = random_failure(nominal, rng, kind=kind)
        k = int(min(n, failure.k0 + rng.integers(*cfg["since_layers"]) + 1))
    else:
        k = int(rng.integers(max(6, n // 10), n + 1))
    printed = printed_stack(nominal, failure, k)
    plan = nominal[:k]

    cam = base_camera(cfg)
    if cfg["camera_jitter"]:
        e = cfg.get("camera_error", 1.0)          # x the usual calibration error (stress tests)
        cam = cam.jittered(rng, pos_mm=2.0 * e, rot_deg=0.8 * e, f_rel=0.02 * e, k1=0.02 * e, k2=0.01 * e)
    drop = float(rng.uniform(0.0, 0.4))
    bounds = stack_bounds(plan)
    top_z = plan[-1].z
    xy, yaw = sample_placement(rng, cam, bounds, top_z, V.plate_z_mm(cam, top_z, drop), cfg)
    look, look_lab = sample_look(rng, cam, xy, 0.5 * np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    bulge = float(rng.uniform(0.02, 0.08))

    # current frame (after layer k) and what the plan would look like there
    s = _split(printed, plan)
    meshes = {"base": V.stack_mesh(printed, 0, s, bulge), "printed": V.stack_mesh(printed, s, None, bulge),
              "plan": V.stack_mesh(plan, s, None, bulge)}
    with V.P1SScene(cam, look, meshes, xy, yaw, top_z, drop) as scene:
        cur, d_print = scene.render(["base", "printed"])
        _, d_plan = scene.render(["base", "plan"])
    mask = clean_mask(np.abs(d_print - d_plan) > CHANGE_MM)
    # previous frame: one layer earlier, so the bed is one layer higher
    prev_stack = printed[:k - 1] if k > 1 else printed
    with V.P1SScene(cam, look, {"p": V.stack_mesh(prev_stack, bulge_mm=bulge)}, xy, yaw, prev_stack[-1].z, drop) as scene:
        prev, _ = scene.render(["p"])
    # the guide is drawn exactly as on the printer: from the calibration (not the slightly different
    # camera that took the frames), from the sliced layers as planned, with no extra bed drop
    guide = plan_silhouette(base_camera(cfg), look, nominal0[:k], xy, yaw)
    ph0 = V.Photometric.sample(rng)
    ph1 = ph0.perturbed(rng)

    ys, xs = np.nonzero(mask)
    label = dict(id=f"{i:06d}", split="val" if cfg["val_every"] and i % cfg["val_every"] == 0 else "train",
                 part=part.name, kind=kind, layer=k, layers_total=n, top_z_mm=round(top_z, 3),
                 failure_started_layer=None if failure is None else int(failure.k0),
                 severity=None if failure is None else round(float(failure.p.get("severity", 0.0)), 3),
                 part_xy_mm=[round(v, 2) for v in xy], part_yaw_deg=round(yaw, 2), bed_drop_mm=round(drop, 3),
                 part_bounds_mm=[round(v, 3) for v in bounds], change_px=int(len(xs)),
                 change_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                 visible=bool(kind != "ok" and len(xs) >= 15),
                 look=dict(**look_lab, n_lights=len(look.lights), toolhead_in_scene=look.toolhead_xy is not None),
                 camera=cam.to_json())
    return prev, cur, guide, mask, label, (ph0, ph1)


def regen_guides(data_dir, calib=None, workers=12, log=print):
    """Redraw every guide of an existing dataset from the calibration (as `make_sample` now does)."""
    data_dir = Path(data_dir)
    labels = [json.loads(l) for l in open(data_dir / "labels.jsonl")]
    jobs = [(str(data_dir), l, calib) for l in labels]
    with mp.get_context("spawn").Pool(safe_workers(workers, log), maxtasksperchild=TASKS_PER_WORKER * 4) as pool:
        for k, _ in enumerate(pool.imap_unordered(_regen_one, jobs, chunksize=8)):
            if (k + 1) % 500 == 0:
                log(f"{k + 1}/{len(jobs)} guides")
    return len(jobs)


def _regen_one(args):
    data_dir, l, calib = args
    if l["part"].startswith("shape-"):
        path = f"shape:{int(l['part'].split('-')[1])}"
    else:
        paths = default_parts() + glob.glob("tests/fixtures/unseen_parts/*.yaml")
        path = next(p for p in paths if _nominal(p)[0].name == l["part"])
    _, nominal = _nominal(path)
    cam = V.P1SCamera.load(calib) if calib else V.P1SCamera.default()
    g = plan_silhouette(cam, V.Look(), nominal[:l["layer"]], l["part_xy_mm"], l["part_yaw_deg"])
    cv2.imwrite(str(Path(data_dir) / l["files"]["guide"]), g)


def _write(i, cfg, out):
    lab_path = out / "labels" / f"{i:06d}.json"
    if lab_path.exists():
        return i, "skip"
    prev, cur, guide, mask, label, (ph0, ph1) = make_sample(i, cfg)
    rng = np.random.default_rng([cfg["seed"], i, 7])
    names = dict(prev=f"images/{i:06d}_prev.jpg", cur=f"images/{i:06d}_cur.jpg",
                 guide=f"guides/{i:06d}.png", mask=f"masks/{i:06d}.png")
    for key, rgb, ph in (("prev", prev, ph0), ("cur", cur, ph1)):
        cv2.imwrite(str(out / names[key]), ph.apply(rgb, rng, jpeg=False)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, ph.jpeg_quality])
    cv2.imwrite(str(out / names["guide"]), guide)
    cv2.imwrite(str(out / names["mask"]), mask)
    label["files"] = names
    tmp = lab_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(label))
    tmp.replace(lab_path)
    return i, label["kind"]


def _worker(args):
    i, cfg, out = args
    try:
        return _write(i, cfg, Path(out))
    except Exception as e:
        return i, f"error: {type(e).__name__}: {e}"


def generate(n, out, cfg, workers=1, start=0, log=print):
    out = Path(out)
    for d in ("images", "guides", "masks", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    jobs = [(i, cfg, str(out)) for i in range(start, start + n)]
    t0, done, errors = time.time(), 0, []
    workers = safe_workers(workers, log)
    pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=TASKS_PER_WORKER) if workers > 1 else None
    it = pool.imap_unordered(_worker, jobs, chunksize=4) if pool else map(_worker, jobs)
    try:
        for i, status in it:
            done += 1
            if status.startswith("error"):
                errors.append(dict(id=i, error=status))
            if done % 100 == 0 or done == n:
                rate = done / (time.time() - t0)
                log(f"{done}/{n}  {rate:.1f}/s  eta {(n - done) / max(rate, 1e-9) / 60:.1f} min  errors {len(errors)}")
    finally:
        if pool:
            pool.close()
            pool.join()
    labels = [json.loads(p.read_text()) for p in sorted((out / "labels").glob("*.json"))]
    with open(out / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    manifest = dict(generator="recovery.p1s_layer_data", git_commit=_git_commit(), argv=sys.argv, config=cfg,
                    base_camera=base_camera(cfg).to_json(), samples=len(labels), errors=errors,
                    seconds=round(time.time() - t0, 1), kinds={k: sum(l["kind"] == k for l in labels) for k in KINDS},
                    visible={k: sum(l["kind"] == k and l["visible"] for l in labels) for k in FAILURES},
                    parts={p: sum(l["part"] == p for l in labels) for p in sorted({l["part"] for l in labels})})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/p1s_layer")
    ap.add_argument("--workers", type=int, help="default: as many as the CPUs and free RAM allow")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="*", help="part YAMLs (default: every part in parts/)")
    ap.add_argument("--calib", help="camera calibration .npz (needs calib_z_mm)")
    ap.add_argument("--shapes", type=int, default=160, help="random training shapes to add (0 = real parts only)")
    ap.add_argument("--camera-error", type=float, default=1.0,
                    help="scale of the random calibration error (1 = as in training; 2 = a test with twice as much)")
    ap.add_argument("--regen-guides", metavar="DIR", help="only redraw the guides of an existing dataset")
    a = ap.parse_args(argv)
    if a.regen_guides:
        print(f"{regen_guides(a.regen_guides, a.calib, a.workers)} guides redrawn")
        return 0
    cfg = default_config(seed=a.seed, calib=a.calib, camera_error=a.camera_error, shapes=a.shapes,
                         **({"parts": a.parts} if a.parts else {}))
    m = generate(a.n, a.out, cfg, a.workers, a.start, log=lambda s: print(s, flush=True))
    print(json.dumps({k: m[k] for k in ("samples", "kinds", "visible", "parts", "seconds")}))
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
