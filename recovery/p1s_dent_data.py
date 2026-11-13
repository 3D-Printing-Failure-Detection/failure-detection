"""Synthetic training pairs for dent / poke detection from the P1S chamber camera.

Each sample is the real dent test (parts/print/water-tray/README.md) done in simulation:
1. print the part to about half height (the pause layer; the water tray pauses at layer 30, z = 6 mm);
2. render a BEFORE frame from the P1S camera (recovery/p1s_view.py: front-left, near-horizontal fisheye,
   the bed lowered so the top layer is at the nozzle plane);
3. change the warm top layers as a person does during the pause, with `dent_layers` (gentle push, the
   wall bends in) or `poke_layers` (tool pushed through the wall) from recovery/agent/rehearse.py,
   or leave it (`none`, the negative class);
4. render an AFTER frame from the same camera, lighting and bed position, with fresh sensor noise.

Labels per pair: kind (dent / poke / none), wall (front / back / left / right of the part as printed:
front = -Y, nearest the door), position along the wall, depth, the tool size, whether the cavity is
open afterwards (it would leak), the dent centre in pixels, and a pixel mask of what changed (from the
before/after depth images). Appearance is randomised: filament colour and gloss, plate type, chamber
colour, lights, toolhead in or out of view, part placement and yaw, pause layer, camera pose and lens
within the calibration's uncertainty, exposure, white balance, blur, noise and JPEG quality.

    python -m recovery.p1s_dent_data --n 3000 --out data/p1s_dent --workers 12
    python -m recovery.p1s_dent_data --n 3000 --out data/p1s_dent --calib calib.npz   # a real calibration

Real dent-test photos are the test set: label them in the same labels.jsonl format.
"""
from . import p1s_view as V                 # first: selects the EGL backend before MuJoCo loads

import argparse
import colorsys
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point

from .agent.rehearse import _push, dent_layers, poke_layers
from .geometry.ops import clean, polys
from .parts import get_part
from .types import Layer

WALLS = ("front", "back", "left", "right")    # part frame: front -Y, back +Y, left -X, right +X
KINDS = ("dent", "poke", "none")
DEFAULT_PART = "parts/print/water-tray/water-tray.yaml"
MESH_SPLIT = 10            # the top MESH_SPLIT layers are meshed separately (before / after versions)
CHANGE_MM = 0.05           # depth change that counts as "changed" in the mask

FILAMENTS = {              # common PLA colours (linear-ish RGB), jittered per sample
    "white": (0.92, 0.92, 0.90), "black": (0.06, 0.06, 0.07), "grey": (0.50, 0.51, 0.53),
    "red": (0.80, 0.10, 0.10), "blue": (0.10, 0.25, 0.75), "green": (0.15, 0.60, 0.25),
    "yellow": (0.95, 0.80, 0.10), "orange": (0.95, 0.45, 0.10), "purple": (0.45, 0.20, 0.65),
    "pink": (0.95, 0.55, 0.70), "beige": (0.85, 0.75, 0.60), "silver": (0.70, 0.71, 0.74),
}
PLATES = {                 # (speckle colour 1, colour 2, specular)
    "textured_pei": ((0.45, 0.36, 0.20), (0.30, 0.24, 0.13), 0.15),
    "smooth_pei": ((0.36, 0.28, 0.14), (0.30, 0.23, 0.12), 0.45),
    "cool_plate": ((0.10, 0.10, 0.11), (0.06, 0.06, 0.07), 0.25),
    "engineering": ((0.30, 0.30, 0.31), (0.22, 0.22, 0.23), 0.30),
}


# ---------------------------------------------------------------- geometry
def part_dims(layers):
    x0, y0, x1, y1 = layers[-1].region.bounds
    return x1 - x0, y1 - y0, (x0, y0, x1, y1)


def wall_point(wall, along_mm, bounds, out_mm=0.5):
    """A point just outside the middle of the outer face of `wall` (part frame), `along_mm` from the
    wall's centre (along +X for front/back, +Y for left/right)."""
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"front": (cx + along_mm, y0 - out_mm), "back": (cx + along_mm, y1 + out_mm),
            "left": (x0 - out_mm, cy + along_mm), "right": (x1 + out_mm, cy + along_mm)}[wall]


def _round_and_wobble(layers, rng, corner_mm, wobble_mm):
    out = []
    for lay in layers:
        g = lay.region
        if corner_mm > 0.05:
            g = clean(g.buffer(-corner_mm, quad_segs=4).buffer(corner_mm, quad_segs=4))
        if wobble_mm > 0:
            g = translate(g, *rng.normal(0, wobble_mm, 2))
        out.append(Layer(lay.z, g))
    return out


def cavity_open(layers, z_floor):
    """True if any layer above the floor has lost its closed inner wall (the cavity reaches the outside)."""
    for lay in layers:
        if lay.z <= z_floor + 1e-6:
            continue
        if not any(len(p.interiors) for p in polys(lay.region)):
            return True
    return False


def apply_change(printed, kind, rng, dims, cfg):
    """Pick and apply a person's change to the top layers. Returns (after_layers, label dict)."""
    L, Wd, bounds = dims
    if kind == "none":
        return list(printed), dict(kind="none", wall=None)
    wall = WALLS[rng.integers(len(WALLS))]
    half = (L if wall in ("front", "back") else Wd) / 2
    along = float(rng.uniform(-0.75, 0.75) * half)
    x, y = wall_point(wall, along, bounds)
    c, n0 = _push(printed[-1], x, y)
    if kind == "dent":
        taper = int(rng.integers(3, 9))
        depth = float(rng.uniform(*cfg["dent_depth_mm"]))
        width = float(rng.uniform(*cfg["dent_width_mm"]))
        new = dent_layers(printed[-taper:], x, y, depth, width)
        lab = dict(depth_mm=round(depth, 3), width_mm=round(width, 3), layers=taper)
    else:
        k = int(rng.integers(2, 6))
        hole = float(rng.uniform(*cfg["poke_hole_mm"]))
        reach = float(rng.uniform(*cfg["poke_reach_mm"]))
        new = poke_layers(printed[-k:], x, y, hole, reach)
        lab = dict(depth_mm=round(reach, 3), hole_mm=round(hole, 3), layers=k)
    after = printed[:len(printed) - len(new)] + list(new)
    return after, dict(kind=kind, wall=wall, along_mm=round(along, 3), contact_mm=np.round(c, 3).tolist(),
                       inward=np.round(n0, 4).tolist(), **lab)


def clean_mask(changed, min_px=4):
    """Depth-difference mask without isolated pixels (depth-buffer noise on silhouettes), uint8 0/255."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    keep = np.r_[False, stats[1:, cv2.CC_STAT_AREA] >= min_px]
    return keep[lab].astype(np.uint8) * 255


# ---------------------------------------------------------------- appearance / placement
def _jitter_rgb(rgb, rng):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    h = (h + rng.normal(0, 0.02)) % 1
    l = float(np.clip(l * rng.uniform(0.85, 1.15), 0.02, 0.97))
    s = float(np.clip(s * rng.uniform(0.8, 1.1), 0, 1))
    return tuple(round(v, 4) for v in colorsys.hls_to_rgb(h, l, s))


def _corners_world(bounds, xy, yaw, z_lo, z_hi):
    x0, y0, x1, y1 = bounds
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    pts = []
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        wx, wy = c * px - s * py + xy[0], s * px + c * py + xy[1]
        pts += [(wx, wy, z_lo), (wx, wy, z_hi)]
    return np.array(pts)


def to_world(p, xy, yaw, z):
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([c * p[0] - s * p[1] + xy[0], s * p[0] + c * p[1] + xy[1], z])


def sample_placement(rng, cam, bounds, top_z, plate_z, cfg, margin_px=24):
    """Part centre on the bed and yaw such that the whole part is in the image."""
    for _ in range(200):
        xy = cfg["part_xy"] or (float(rng.uniform(*cfg["bed_range_mm"])), float(rng.uniform(*cfg["bed_range_mm"])))
        yaw = float(rng.normal(0, cfg["yaw_sigma_deg"]))
        if rng.random() < cfg["yaw_flip_p"]:
            yaw += 180.0
        px = cam.project(_corners_world(bounds, xy, yaw, plate_z, plate_z + top_z))
        w, h = cam.size
        if (px[:, 0].min() > margin_px and px[:, 0].max() < w - margin_px and px[:, 1].min() > margin_px
                and px[:, 1].max() < h - margin_px):
            return xy, yaw
        if cfg["part_xy"]:
            raise ValueError(f"part at {cfg['part_xy']} is not fully in the camera view")
    raise RuntimeError("could not place the part in view")


def sample_look(rng, cam, xy, footprint_r):
    nozzle_z = cam.calib_z_mm
    fil = list(FILAMENTS)[rng.integers(len(FILAMENTS))]
    plate = list(PLATES)[rng.integers(len(PLATES))]
    p1, p2, spec = PLATES[plate]
    lights = []
    for k in range(int(rng.integers(1, 4))):
        pos = (float(rng.uniform(-20, 280)), float(rng.uniform(-20, 280)), nozzle_z + float(rng.uniform(40, 220)))
        tgt = (xy[0] + float(rng.normal(0, 30)), xy[1] + float(rng.normal(0, 30)), nozzle_z - 10.0)
        lights.append(dict(pos=pos, target=tgt, diffuse=float(rng.uniform(0.5, 1.0) / (1 + 0.5 * k)),
                           specular=float(rng.uniform(0.0, 0.4)), spot=bool(rng.random() < 0.7), shadow=False))
    toolhead = None
    if rng.random() < 0.4:
        # parked somewhere on the bed plane but not between the camera and the part
        cxy = cam.centre[:2]
        for _ in range(50):
            t = np.array([rng.uniform(0, 256), rng.uniform(0, 256)])
            d = np.asarray(xy) - cxy
            s = np.clip(np.dot(t - cxy, d) / np.dot(d, d), 0, 1.3)
            if np.linalg.norm(t - (cxy + s * d)) > footprint_r + 40 and np.linalg.norm(t - xy) > footprint_r + 40:
                toolhead = (float(t[0]), float(t[1]))
                break
    g = float(rng.uniform(0.05, 0.25))
    look = V.Look(filament_rgb=_jitter_rgb(FILAMENTS[fil], rng), filament_specular=float(rng.uniform(0.05, 0.6)),
                  filament_shininess=float(rng.uniform(0.1, 0.8)),
                  plate_rgb1=_jitter_rgb(p1, rng), plate_rgb2=_jitter_rgb(p2, rng), plate_specular=spec,
                  chamber_rgb=(g, g * rng.uniform(0.95, 1.05), g * rng.uniform(0.95, 1.1)),
                  ambient=float(rng.uniform(0.15, 0.5)), headlight=float(rng.uniform(0.05, 0.35)),
                  lights=lights, toolhead_xy=toolhead,
                  toolhead_rgb=tuple(float(v) for v in rng.uniform(0.08, 0.3, 3)))
    return look, dict(filament=fil, plate=plate)


# ---------------------------------------------------------------- one pair
def default_config(**kw):
    cfg = dict(part=DEFAULT_PART, kinds=KINDS, weights=(0.4, 0.3, 0.3), pause_jitter=3,
               dent_depth_mm=(0.4, 3.0), dent_width_mm=(2.5, 6.0), poke_hole_mm=(1.0, 3.0), poke_reach_mm=(1.6, 4.0),
               bed_range_mm=(40.0, 140.0), part_xy=None, yaw_sigma_deg=8.0, yaw_flip_p=0.15,
               camera_jitter=True, calib=None, seed=0, val_every=10)
    cfg.update(kw)
    return cfg


_CACHE = {}


def _nominal(part_path):
    if part_path not in _CACHE:
        part = get_part(part_path)
        _CACHE[part_path] = (part, part.layers())
    return _CACHE[part_path]


def base_camera(cfg):
    return V.P1SCamera.load(cfg["calib"]) if cfg["calib"] else V.P1SCamera.default()


def make_pair(i, cfg):
    """Render sample i. Returns (before_rgb, after_rgb, mask uint8, label dict, (photo_before, photo_after))."""
    rng = np.random.default_rng([cfg["seed"], i])
    part, nominal = _nominal(cfg["part"])
    n = len(nominal)
    pause = int(np.clip(n // 2 + rng.integers(-cfg["pause_jitter"], cfg["pause_jitter"] + 1), MESH_SPLIT + 1, n))
    printed = _round_and_wobble(nominal[:pause], rng, float(rng.uniform(0.0, 0.4)), float(rng.uniform(0.0, 0.03)))
    dims = part_dims(printed)
    kind = cfg["kinds"][rng.choice(len(cfg["kinds"]), p=np.asarray(cfg["weights"]) / np.sum(cfg["weights"]))]
    after, change = apply_change(printed, kind, rng, dims, cfg)

    cam = base_camera(cfg)
    if cfg["camera_jitter"]:
        cam = cam.jittered(rng)
    top_z = printed[-1].z
    drop = float(rng.uniform(0.0, 0.6))
    plate_z = V.plate_z_mm(cam, top_z, drop)
    xy, yaw = sample_placement(rng, cam, dims[2], top_z, plate_z, cfg)
    look, look_lab = sample_look(rng, cam, xy, 0.5 * np.hypot(dims[0], dims[1]))
    bulge = float(rng.uniform(0.02, 0.08))

    split = len(printed) - MESH_SPLIT
    meshes = {"base": V.stack_mesh(printed, 0, split, bulge),
              "before": V.stack_mesh(printed, split, None, bulge),
              "after": V.stack_mesh(after, split, None, bulge)}
    with V.P1SScene(cam, look, meshes, xy, yaw, top_z, drop) as scene:
        rgb0, d0 = scene.render(["base", "before"])
        rgb1, d1 = scene.render(["base", "after"])
    mask = clean_mask(np.abs(d1 - d0) > CHANGE_MM)
    ph0 = V.Photometric.sample(rng)
    ph1 = ph0.perturbed(rng)

    label = dict(id=f"{i:06d}", split="val" if cfg["val_every"] and i % cfg["val_every"] == 0 else "train",
                 part=part.name, pause_layer=pause, top_z_mm=round(top_z, 3), bed_drop_mm=round(drop, 3),
                 part_xy_mm=[round(v, 2) for v in xy], part_yaw_deg=round(yaw, 2), **change)
    ys, xs = np.nonzero(mask)
    label["change_px"] = int(len(xs))
    label["change_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    if change["kind"] != "none":
        c_world = to_world(change["contact_mm"], xy, yaw, plate_z + top_z - 0.2)
        label["contact_px"] = np.round(cam.project([c_world])[0], 1).tolist()
        n_world = to_world(-np.asarray(change["inward"]), (0, 0), yaw, 0)[:2]
        label["outer_face_toward_camera"] = bool(np.dot(n_world, cam.centre[:2] - c_world[:2]) > 0)
        label["visible"] = label["change_px"] >= 15
    else:
        label["visible"] = False
    floor = float(part.spec.get("container", {}).get("base_h", 0.0))
    label["cavity_open_after"] = cavity_open(after[-MESH_SPLIT:], floor) and not cavity_open(printed[-MESH_SPLIT:], floor)
    label.update(look=dict(**look_lab, filament_rgb=list(look.filament_rgb), n_lights=len(look.lights),
                           toolhead_in_scene=look.toolhead_xy is not None, layer_bulge_mm=round(bulge, 3)),
                 photometric=dict(gain=round(ph0.gain, 3), gamma=round(ph0.gamma, 3), blur=round(ph0.blur_sigma, 3),
                                  noise=round(ph0.noise_std, 2), jpeg=ph0.jpeg_quality),
                 camera=cam.to_json())
    return rgb0, rgb1, mask, label, (ph0, ph1)


def _write(i, cfg, out):
    lab_path = out / "labels" / f"{i:06d}.json"
    if lab_path.exists():
        return i, "skip"
    rgb0, rgb1, mask, label, (ph0, ph1) = make_pair(i, cfg)
    rng = np.random.default_rng([cfg["seed"], i, 7])
    names = dict(before=f"images/{i:06d}_before.jpg", after=f"images/{i:06d}_after.jpg", mask=f"masks/{i:06d}.png")
    for key, rgb, ph in (("before", rgb0, ph0), ("after", rgb1, ph1)):
        img = ph.apply(rgb, rng, jpeg=False)
        cv2.imwrite(str(out / names[key]), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, ph.jpeg_quality])
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
    except Exception as e:                   # keep going; failures are listed in the manifest
        return i, f"error: {type(e).__name__}: {e}"


def contact_sheet(out, ids, path, cols=4, scale=0.3):
    """before | after | change-mask overlay for a few pairs, for a quick look."""
    rows = []
    for i in ids:
        lab = json.loads((out / "labels" / f"{i:06d}.json").read_text())
        b, a = cv2.imread(str(out / lab["files"]["before"])), cv2.imread(str(out / lab["files"]["after"]))
        m = cv2.imread(str(out / lab["files"]["mask"]), cv2.IMREAD_GRAYSCALE)
        ov = a.copy()
        ov[m > 0] = (0.4 * ov[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = lab["change_bbox"] or (0, 0, 0, 0)
        tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in (b, a, ov)]
        txt = f"{lab['id']} {lab['kind']} {lab.get('wall') or ''} d={lab.get('depth_mm', 0)} px={lab['change_px']}"
        cv2.putText(tiles[0], txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


def generate(n, out, cfg, workers=1, start=0, log=print):
    out = Path(out)
    for d in ("images", "masks", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    jobs = [(i, cfg, str(out)) for i in range(start, start + n)]
    t0, done, errors, counts = time.time(), 0, [], {}
    if workers > 1:
        pool = mp.get_context("spawn").Pool(workers)
        it = pool.imap_unordered(_worker, jobs, chunksize=4)
    else:
        pool, it = None, map(_worker, jobs)
    try:
        for i, status in it:
            done += 1
            if status.startswith("error"):
                errors.append(dict(id=i, error=status))
            counts[status if not status.startswith("error") else "error"] = counts.get(status, 0) + 1
            if done % 50 == 0 or done == n:
                rate = done / (time.time() - t0)
                log(f"{done}/{n}  {rate:.1f} pairs/s  eta {(n - done) / max(rate, 1e-9) / 60:.1f} min  {counts}")
    finally:
        if pool:
            pool.close()
            pool.join()
    labels = [json.loads(p.read_text()) for p in sorted((out / "labels").glob("*.json"))]
    with open(out / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    summary = dict(pairs=len(labels), errors=errors, seconds=round(time.time() - t0, 1),
                   kinds={k: sum(l["kind"] == k for l in labels) for k in KINDS},
                   visible={k: sum(l["kind"] == k and l["visible"] for l in labels) for k in ("dent", "poke")},
                   splits={s: sum(l["split"] == s for l in labels) for s in ("train", "val")})
    cam = base_camera(cfg)
    manifest = dict(generator="recovery.p1s_dent_data", git_commit=_git_commit(), argv=sys.argv,
                    config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                    base_camera=cam.to_json(), base_camera_source=cfg["calib"] or "p1s_view.DEFAULT_CAMERA (issue-5 clicks)",
                    image_size=list(cam.size), change_threshold_mm=CHANGE_MM, **summary)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=3000, help="number of pairs")
    ap.add_argument("--out", default="data/p1s_dent")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--start", type=int, default=0, help="first sample index (to extend a dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default=DEFAULT_PART, help="part YAML (default: the water tray)")
    ap.add_argument("--calib", help="camera calibration .npz from calibrate-bed / calibrate-board (needs calib_z_mm)")
    ap.add_argument("--no-camera-jitter", action="store_true", help="use the calibration exactly")
    ap.add_argument("--part-xy", help="fix the part centre on the bed, 'X,Y' mm from the front-left plate corner")
    ap.add_argument("--weights", default="0.4,0.3,0.3", help="sampling weights for dent,poke,none")
    ap.add_argument("--preview", type=int, default=12, help="pairs in preview.jpg (0: none)")
    a = ap.parse_args(argv)
    cfg = default_config(part=a.part, calib=a.calib, seed=a.seed, camera_jitter=not a.no_camera_jitter,
                         weights=tuple(float(w) for w in a.weights.split(",")),
                         part_xy=tuple(float(v) for v in a.part_xy.split(",")) if a.part_xy else None)
    m = generate(a.n, a.out, cfg, a.workers, a.start, log=lambda s: print(s, flush=True))
    print(json.dumps({k: m[k] for k in ("pairs", "kinds", "visible", "splits", "seconds")}))
    if m["errors"]:
        print(f"{len(m['errors'])} samples failed; see manifest.json", file=sys.stderr)
    if a.preview:
        ids = [int(l.split('"id": "')[1][:6]) for l in open(Path(a.out) / "labels.jsonl")][: a.preview]
        contact_sheet(Path(a.out), ids, Path(a.out) / "preview.jpg")
        print(f"preview: {Path(a.out) / 'preview.jpg'}")
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
