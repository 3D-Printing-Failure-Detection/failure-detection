"""python -m recovery.cli <command>

  demo      run one simulated failure through the loop, save figure + G-code
  eval      simulation evaluation vs baselines   (same as python -m recovery.eval.run_eval)
  gen-data  generate synthetic training data
  train     train detector + segmenter
  app       launch the Gradio interface
  specgen   LLM (or heuristic) pins form + function of ANY part into a verified spec
  stress    FE stress-test the part about to be printed (optionally with an injected failure)
  slice     cross-section an STL into layers and write G-code (no slicer needed)
  calibrate calibrate the P1S chamber camera against a checkerboard
  live      run the loop on a real Bambu P1S (print started from Bambu Studio)
"""
import argparse
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Delegate before argparse consumes the child command's flags.
    if argv and argv[0] in {"eval", "mujoco-eval", "gen-data", "train"}:
        if argv[0] == "gen-data":
            import runpy
            sys.argv = ["synth_data"] + argv[1:]
            runpy.run_module("recovery.models.synth_data", run_name="__main__")
            return
        import importlib
        module = {"eval": ".eval.run_eval", "mujoco-eval": ".eval.mujoco_eval",
                  "train": ".models.train"}[argv[0]]
        return importlib.import_module(module, "recovery").main(argv[1:])
    ap = argparse.ArgumentParser(prog="recovery")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo")
    d.add_argument("--part", required=True, help="part name in parts/ or a path to a part YAML")
    d.add_argument("--kind", default="collapse")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--severity", type=float, default=None)
    d.add_argument("--models", default="classical", choices=["classical", "torch"])
    d.add_argument("--policy", default="ours", choices=["ours", "alert", "functional", "guarded"],
                   help="functional: experimental measured-state and predictive recovery")
    d.add_argument("--out", default="results/demo")

    for name in ("eval", "mujoco-eval", "gen-data", "train"):
        p = sub.add_parser(name, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
    sub.add_parser("app")
    s = sub.add_parser("slice")
    s.add_argument("stl")
    s.add_argument("--layer-h", type=float, default=0.2)
    s.add_argument("--out", default="results/sliced.gcode")

    sg = sub.add_parser("specgen")
    sg.add_argument("--part", required=True)
    sg.add_argument("--describe", default="", help="plain-English description of the part's job")
    sg.add_argument("--no-llm", action="store_true", help="deterministic heuristic author only")
    sg.add_argument("--out", default=None, help="default: parts/specs/<name>.yaml")
    st = sub.add_parser("stress")
    st.add_argument("--part", required=True)
    st.add_argument("--h", type=float, default=1.0, help="voxel size in mm (smaller = finer, slower)")
    st.add_argument("--failure", default=None, help="inject this failure and stress-test the damaged part")
    st.add_argument("--severity", type=float, default=None)
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--out", default="results/stress")

    c = sub.add_parser("calibrate")
    c.add_argument("--ip", required=True)
    c.add_argument("--code", required=True, help="printer LAN access code")
    c.add_argument("--out", default="calib.npz")
    c.add_argument("--square-mm", type=float, default=10.0)
    c.add_argument("--board-origin", type=float, nargs=2, default=[100.0, 100.0],
                   help="bed X Y (mm) of the board's first inner corner in the flat shot")
    lv = sub.add_parser("live")
    lv.add_argument("--part", required=True, help="part YAML with `gcode:` pointing at the sliced .gcode.3mf")
    lv.add_argument("--ip", required=True)
    lv.add_argument("--serial", required=True)
    lv.add_argument("--code", required=True)
    lv.add_argument("--calib", default="calib.npz")
    lv.add_argument("--models", default="classical", choices=["classical", "torch"])
    lv.add_argument("--mode", default="restart", choices=["restart", "stream"])
    lv.add_argument("--rehome-gcode", default="")
    lv.add_argument("--skip-stress", action="store_true", help="do not gate the print on the FE stress test")

    a = ap.parse_args(argv)
    if a.cmd == "specgen":
        from .parts import get_part
        from .specgen import author_spec, save_spec
        part = get_part(a.part)
        res = author_spec(part.layers(), part.layer_h, a.describe or part.spec.get("description", ""),
                          use_llm=not a.no_llm)
        out = Path(a.out or f"parts/specs/{part.name}.yaml")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_spec(res, out, part.name)
        print(f"source: {res['source']}  verified against the nominal part: {res['verified']}")
        print(res["rationale"])
        for i in res["issues"]:
            print("  issue:", i)
        print("wrote", out, "-> reference it from the part YAML with  spec_file:", out.name)
        return
    if a.cmd == "stress":
        import numpy as np
        from .deform import random_failure
        from .fea import stress_test
        from .fea.viz import stress_figure
        from .parts import get_part
        part = get_part(a.part)
        layers = part.layers()
        title = part.name
        if a.failure:
            f = random_failure(layers, np.random.default_rng(a.seed), kind=a.failure, severity=a.severity)
            layers = [f.apply(l, i) if f.active(i) else l for i, l in enumerate(layers)]
            title += f" with {a.failure} from layer {f.k0}"
        rep = stress_test(layers, part.spec, part.layer_h, a.h)
        print(title)
        print(rep.table() if rep.cases else "(no `loads:` in this part's spec; run specgen first)")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(rep.cases):
            if case.fi3 is not None:
                stress_figure(case, title).savefig(out / f"{part.name}_case{i}.png", dpi=140)
        print("verdict:", "PASS: safe to print" if rep.ok else "FAIL: " + "; ".join(rep.violations))
        return
    if a.cmd == "calibrate":
        from .live import calibrate
        calibrate(a.ip, a.code, a.out, square_mm=a.square_mm, board_origin=tuple(a.board_origin))
        return
    if a.cmd == "live":
        from .live import run_live
        run_live(a.part, a.ip, a.serial, a.code, a.calib, a.models, a.mode, a.rehome_gcode, a.skip_stress)
        return
    if a.cmd == "demo":
        from .run import run_scenario
        from .viz import event_figure
        sc = run_scenario(a.part, a.kind, a.seed, a.severity, models=a.models, policy=a.policy, log=print)
        res = sc.result
        print(f"injected {sc.failure.kind} at layer {sc.failure.k0}; if ignored: {sc.ignore_verdict}")
        print(f"outcome: {res.outcome}; final: {res.final}")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "final.gcode").write_text(sc.gcode)
        if res.events and sc.printed:
            event_figure(sc.nominal, res.events[0], sc.printed, f"{a.part}: {a.kind}") \
                .savefig(out / "event.png", dpi=150)
            print("wrote", out / "event.png", out / "final.gcode")
    elif a.cmd == "eval":
        from .eval.run_eval import main as m
        m(a.rest)
    elif a.cmd == "gen-data":
        sys.argv = ["synth_data"] + a.rest
        import runpy
        runpy.run_module("recovery.models.synth_data", run_name="__main__")
    elif a.cmd == "train":
        from .models.train import main as m
        m(a.rest)
    elif a.cmd == "app":
        from .app import build
        build().launch()
    elif a.cmd == "slice":
        from .gcode import write_gcode
        from .slicing import stl_to_layers
        layers = stl_to_layers(a.stl, a.layer_h)
        text, fil = write_gcode(layers, a.layer_h)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"{len(layers)} layers, {fil:.0f} mm filament -> {a.out}")


if __name__ == "__main__":
    main()
