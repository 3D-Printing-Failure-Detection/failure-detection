"""End-to-end check of the live path: the simulated P1S (recovery/printer/fake_p1s.py) prints, and `watch`
reads it over the same connections as a real printer (status over MQTT, camera on port 6000), checks every
layer with the trained model and applies the agent's alarm rule. One print per scenario; reports whether
and when it alarmed.

    python -m recovery.eval.live_sim_check --ckpt runs/p1s_layer --out results/live_sim_check
"""
from .. import p1s_view as V  # noqa: F401  (first: selects the EGL backend before MuJoCo loads)

import argparse
import json
import time
from pathlib import Path

SCENARIOS = [  # part, failure, layers of print before it starts, severity
    ("parts/mug.yaml", None, None, None),
    ("parts/print/water-tray/water-tray.yaml", None, None, None),
    ("parts/mug.yaml", "warp", 15, 3.0),
    ("parts/print/water-tray/water-tray.yaml", "collapse", 15, 4.0),
    ("parts/spacer.yaml", "shift", 15, 3.0),
    ("parts/print/star-knob/star-knob.yaml", "under_extrusion", 15, 1.2),
    ("parts/cup.yaml", "detach", 15, None),
]


def run_one(i, part_path, fail, after, severity, ckpt, out, layer_s, window, seed):
    from ..live import run_watch
    from ..parts import get_part
    from ..printer.fake_p1s import FakeP1S
    part = get_part(part_path)
    n = len(part.layers())
    start = max(0, n // 3 - (after or 15))
    fail_at = start + after if fail else None
    p = FakeP1S(part, mqtt_port=0, camera_port=0, layer_s=layer_s, fail=fail, fail_at=fail_at, severity=severity,
                start_layer=start, seed=seed + i, log=lambda *a: None)
    p.n_stop = min(n, start + window)
    p.nominal_all = p.nominal
    try:
        p.heartbeat().start(wait_for_watcher=True)
        run = run_watch(part_path, None, "127.0.0.1", p.serial, p.code, None, ckpt, out=str(Path(out) / f"{i:02d}"),
                        ports=p.ports, log=lambda *a: None, stop_after=p.n_stop)
    finally:
        p.close()
    rows = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines()]
    meta = json.loads((run / "meta.json").read_text())
    alarm = meta["first_alarm"]
    res = dict(part=part.name, failure=fail or "none", failure_starts=fail_at, layers_checked=sum("flagged_px" in r for r in rows),
               layers_unchecked=meta["layers_unchecked"], layers_skipped=meta["layers_skipped"],
               first_alarm=alarm, run=str(run))
    if fail:
        res["result"] = ("caught" if alarm and alarm["layer"] >= fail_at else
                         "false alarm before it started" if alarm else "missed")
        if alarm and alarm["layer"] >= fail_at:
            res["layers_late"] = alarm["layer"] - fail_at + 1
    else:
        res["result"] = "false alarm" if alarm else "no alarm (right)"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="runs/p1s_layer")
    ap.add_argument("--out", default="results/live_sim_check")
    ap.add_argument("--layer-s", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=45, help="layers watched per print")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sc in enumerate(SCENARIOS):
        t0 = time.time()
        r = run_one(i, *sc, a.ckpt, out, a.layer_s, a.window, a.seed)
        r["seconds"] = round(time.time() - t0)
        results.append(r)
        print(f"{r['part']:14s} {r['failure']:16s} -> {r['result']}"
              + (f" ({r['layers_late']} layers after it started)" if "layers_late" in r else "")
              + f"  [{r['layers_checked']} checked, {r['layers_skipped']} skipped, {r['layers_unchecked']} unchecked]",
              flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
