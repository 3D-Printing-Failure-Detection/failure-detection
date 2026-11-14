"""Whole-print test of the camera failure detector: what the agent would do over a real print.

The per-sample numbers (recovery/models/p1s_layer.py eval) score single layers. A print is many layers:
a good print must get through all of them without a false alarm, and a failing one should be caught
soon after it starts. Each simulated print here has its own part, look, camera error and placement
(as in recovery/p1s_layer_data.py), is rendered layer by layer from the P1S camera, and every layer is
checked. Per layer the detector's answer is logged, so alarm rules ("flag N layers in a row") can be
compared afterwards without re-rendering.

    python -m recovery.eval.p1s_print_eval --prints 160 --ckpt runs/p1s_layer --out results/p1s_print_eval
"""
from .. import p1s_view as V                 # first: selects the EGL backend before MuJoCo loads

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from ..alarm import first_alarm
from ..deform import random_failure
from ..p1s_dent_data import _round_and_wobble, base_camera, sample_look, sample_placement
from ..p1s_layer_data import FAILURES, _nominal, default_config, printed_stack, stack_bounds

_W = {}


def _watcher(ckpt):
    if ckpt not in _W:
        from ..models.p1s_layer import LayerWatcher
        _W[ckpt] = LayerWatcher(ckpt)
    return _W[ckpt]


def run_print(i, cfg, ckpt, window=40, after=15):
    rng = np.random.default_rng([cfg["seed"], i])
    path = cfg["parts"][rng.integers(len(cfg["parts"]))]
    part, nominal0 = _nominal(path)
    n = len(nominal0)
    nominal = _round_and_wobble(nominal0, rng, float(rng.uniform(0.0, 0.3)), float(rng.uniform(0.0, 0.03)))
    failing = i % 2 == 1                                    # half the prints fail
    failure, kind = None, "ok"
    if failing:
        kind = FAILURES[rng.integers(len(FAILURES))]
        failure = random_failure(nominal, rng, kind=kind)
        first = max(2, failure.k0 - 8)
        last = min(n, failure.k0 + after)
    else:
        first = int(rng.integers(2, max(3, n - window)))
        last = min(n, first + window)
    cam = base_camera(cfg).jittered(rng, pos_mm=2.0 * cfg["camera_error"], rot_deg=0.8 * cfg["camera_error"],
                                    f_rel=0.02 * cfg["camera_error"], k1=0.02 * cfg["camera_error"],
                                    k2=0.01 * cfg["camera_error"])
    plan_all = nominal
    bounds = stack_bounds(plan_all[:last])
    xy, yaw = sample_placement(rng, cam, bounds, plan_all[last - 1].z,
                               V.plate_z_mm(cam, plan_all[last - 1].z), cfg)
    look, look_lab = sample_look(rng, cam, xy, 0.5 * np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    bulge = float(rng.uniform(0.02, 0.08))
    photo = V.Photometric.sample(rng)
    w = _watcher(ckpt)
    w.cam = base_camera(cfg)                                # the deployed model knows only the calibration
    layers, prev = [], None
    for k in range(first, last + 1):
        printed = printed_stack(nominal, failure, k)
        with V.P1SScene(cam, look, {"p": V.stack_mesh(printed, bulge_mm=bulge)}, xy, yaw, printed[-1].z) as s:
            rgb, _ = s.render(["p"])
        frame = photo.perturbed(rng).apply(rgb, rng)
        if prev is not None:
            r = w.check(prev, frame, plan_all[:k], xy, yaw)
            failed_yet = failure is not None and k > failure.k0
            layers.append(dict(layer=k - 1, kind=r["kind"], flagged_px=r["flagged_px"], failed=failed_yet))
        prev = frame
    return dict(print=i, part=part.name, true_kind=kind, failure_started=None if failure is None else failure.k0,
                layers=layers, look=look_lab)


def _job(args):
    i, cfg, ckpt = args
    try:
        return run_print(i, cfg, ckpt)
    except Exception as e:
        return dict(print=i, error=f"{type(e).__name__}: {e}")


def alarm_layer(layers, confirm, warmup=3):
    """The layer at which the agent would alarm (recovery/alarm.py), or None."""
    return first_alarm(layers, warmup, confirm)


def summarize(prints, confirms=(1, 2, 3), warmup=3):
    ok = [p for p in prints if "error" not in p]
    out = dict(prints=len(ok), errors=len(prints) - len(ok), rules={})
    for c in confirms:
        clean = [p for p in ok if p["true_kind"] == "ok"]
        fail = [p for p in ok if p["true_kind"] != "ok"]
        false = [p for p in clean if alarm_layer(p["layers"], c, warmup)]
        caught, early, delays, kind_ok, by_kind = [], [], [], [], {}
        for p in fail:
            # an alarm before the failure started is a false alarm on a print that later fails
            # the rule runs over the whole print; an alarm on a layer before the failure started is early
            hit = alarm_layer(p["layers"], c, warmup)
            pre = hit if hit is not None and not hit["failed"] else None
            hit = None if pre is not None else hit
            early.append(pre is not None)
            caught.append(hit is not None and pre is None)
            by_kind.setdefault(p["true_kind"], []).append(hit is not None and pre is None)
            if hit is not None and pre is None:
                delays.append(hit["layer"] - p["failure_started"] + 1)
                kind_ok.append(hit["kind"] == p["true_kind"])
        out["rules"][f"{c} in a row"] = dict(
            good_prints_with_a_false_alarm=f"{len(false)}/{len(clean)}",
            failing_prints_caught=f"{sum(caught)}/{len(fail)}",
            failing_prints_false_alarm_before_it_started=f"{sum(early)}/{len(fail)}",
            caught_by_type={k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_kind.items())},
            layers_from_start_to_alarm_median=float(np.median(delays)) if delays else None,
            layers_from_start_to_alarm_90pct=float(np.percentile(delays, 90)) if delays else None,
            type_right_when_caught=round(float(np.mean(kind_ok)), 3) if kind_ok else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prints", type=int, default=160)
    ap.add_argument("--ckpt", default="runs/p1s_layer")
    ap.add_argument("--out", default="results/p1s_print_eval")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--parts", nargs="*")
    ap.add_argument("--camera-error", type=float, default=1.0)
    ap.add_argument("--rescore", metavar="PRINTS_JSONL", help="only re-score saved prints with the alarm rule")
    a = ap.parse_args(argv)
    if a.rescore:
        prints = [json.loads(l) for l in open(a.rescore)]
        print(json.dumps(summarize(prints)["rules"], indent=1))
        return
    cfg = default_config(seed=a.seed, camera_error=a.camera_error, **({"parts": a.parts} if a.parts else {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(i, cfg, a.ckpt) for i in range(a.prints)]
    prints = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, p in enumerate(pool.imap_unordered(_job, jobs)):
            prints.append(p)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{a.prints} prints ({time.time() - t0:.0f}s)", flush=True)
    prints.sort(key=lambda p: p["print"])
    (out / "prints.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prints))
    s = dict(summarize(prints), ckpt=a.ckpt, seed=a.seed, camera_error=a.camera_error, parts=cfg["parts"],
             seconds=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s["rules"], indent=1))


if __name__ == "__main__":
    main()
