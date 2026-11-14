"""python -m recovery.cli <command>

  demo      run one simulated failure through the loop, save figure + G-code
  eval      simulation evaluation vs baselines   (same as python -m recovery.eval.run_eval)
  gen-data  generate synthetic training data
  train     train detector + segmenter
  app       launch the Gradio interface
  specgen   LLM (or heuristic) pins form + function of ANY part into a verified spec
  fea-verify  check the FE solver against closed-form benchmarks
  stress    FE stress-test the part about to be printed (optionally with an injected failure)
  slice     cross-section an STL into layers and write G-code (no slicer needed)
  connection-test  can the laptop talk to the P1S? network, status, camera, files (+ optional pause/resume)
  check-gcode  offline: what the parser makes of a sliced file (markers, layers, outline rebuild)
  agent     agentic monitoring: pause on alarm, function check, decider recommends, a person approves
  llm-check test the agent's language model (local Qwen / Claude) on two sample alarms
  probe     read-only: P1S status over MQTT + one camera frame (no print commands)
  calibrate calibrate the P1S chamber camera against a checkerboard
  watch     watch-only loop on a real P1S: checks every layer, logs and marks up frames, never commands
  sim-printer  a simulated P1S on this machine (same connections as the real one), to test watch/agent
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
    d.add_argument("--policy", default="ours", choices=["ours", "alert", "functional", "guarded", "contour"],
                   help="functional/guarded/contour: experimental measured-state and predictive recovery")
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
    sg.add_argument("--method", choices=["llm", "heuristic", "template", "manual"], default="llm",
                    help="how the function spec is produced (default: llm)")
    sg.add_argument("--no-llm", action="store_true", help="same as --method heuristic")
    sg.add_argument("--template", default=None, help="for --method template: container|compression|cantilever")
    sg.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="template parameter, repeatable (e.g. --param force_N=80)")
    sg.add_argument("--manual", default=None, help="for --method manual: YAML file holding the spec")
    sg.add_argument("--strict", action="store_true", help="llm method: fail instead of falling back")
    sg.add_argument("--out", default=None, help="default: parts/specs/<name>.yaml")
    st = sub.add_parser("stress")
    st.add_argument("--part", required=True)
    st.add_argument("--h", type=float, default=1.0, help="voxel size in mm (smaller = finer, slower)")
    st.add_argument("--solver", choices=["auto", "calculix", "builtin"], default="auto",
                    help="auto = CalculiX if `ccx` is installed, else built-in")
    st.add_argument("--converge", action="store_true", help="also re-run at several mesh sizes")
    fv = sub.add_parser("fea-verify")
    fv.add_argument("--solver", choices=["auto", "calculix", "builtin"], default="auto")
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
    c.add_argument("--calib-z", type=float, required=True,
                   help="bed Z shown on the printer screen during the flat shot (mm)")
    from .agent.cli import add_arguments as agent_arguments, add_decider_arguments
    agent_arguments(sub.add_parser("agent"))
    add_decider_arguments(sub.add_parser("llm-check"))
    ct = sub.add_parser("connection-test")
    ct.add_argument("--ip"), ct.add_argument("--serial"), ct.add_argument("--code")
    ct.add_argument("--commands", action="store_true", help="also pause and resume a RUNNING print (asks first)")
    ct.add_argument("--mock", action="store_true", help="run the checklist against a fake printer")
    ct.add_argument("--mock-broken", nargs="*", default=[], choices=["network", "status", "camera", "files", "commands"],
                    help="--mock: make these steps fail, to see the hints")
    ct.add_argument("--out", default="results/connection-test")
    cg = sub.add_parser("check-gcode")
    cg.add_argument("file", help="Bambu Studio .gcode.3mf or a plain .gcode")
    cg.add_argument("--plate", type=int, default=1)
    cg.add_argument("--out", default="results/check-gcode", help="where to save the layer preview")
    pb = sub.add_parser("probe")
    pb.add_argument("--ip", required=True)
    pb.add_argument("--serial", required=True)
    pb.add_argument("--code", required=True, help="printer LAN access code")
    pb.add_argument("--out", default="results/probe")
    wt = sub.add_parser("watch")
    src = wt.add_mutually_exclusive_group(required=True)
    src.add_argument("--part", help="part YAML with `gcode:` (and a spec, so decisions are meaningful)")
    src.add_argument("--gcode", help="sliced .gcode.3mf/.gcode directly (no function spec)")
    wt.add_argument("--plate", type=int, default=1)
    wt.add_argument("--ip", required=True)
    wt.add_argument("--serial", required=True)
    wt.add_argument("--code", required=True)
    wt.add_argument("--calib", default="calib.npz", help="camera calibration (default camera if missing)")
    wt.add_argument("--layer-model", default="runs/p1s_layer", help="failure detector checkpoint")
    wt.add_argument("--out", default="results/watch")
    wt.add_argument("--view", type=int, metavar="PORT", help="serve a live page of the marked-up frames on this port")
    wt.add_argument("--view-host", default="127.0.0.1", help="address for the live page (0.0.0.0 = whole network)")
    wt.add_argument("--ports", help="MQTT,camera ports if not 8883,6000 (the simulated printer)")
    sp = sub.add_parser("sim-printer", help="a simulated P1S on this machine (same connections as the real one)")
    ssrc = sp.add_mutually_exclusive_group(required=True)
    ssrc.add_argument("--part", help="part YAML")
    ssrc.add_argument("--gcode", help="sliced .gcode.3mf/.gcode")
    sp.add_argument("--plate", type=int, default=1)
    sp.add_argument("--fail", choices=["collapse", "warp", "shift", "under_extrusion", "detach"],
                    help="make this failure happen (default: a good print)")
    sp.add_argument("--fail-at", type=int, help="layer the failure starts at (default: random)")
    sp.add_argument("--severity", type=float, help="failure size in mm (default: random)")
    sp.add_argument("--layer-s", type=float, default=3.0, help="seconds per layer")
    sp.add_argument("--start-layer", type=int, default=0, help="begin part-way through the print")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--mqtt-port", type=int, default=8883)
    sp.add_argument("--camera-port", type=int, default=6000)
    sp.add_argument("--serial", default="SIMP1S")
    sp.add_argument("--code", default="12345678")
    sp.add_argument("--camera-error", type=float, default=1.0, help="how far the camera is off the calibration")
    sp.add_argument("--calib", help="calibration .npz the simulated camera is close to (default camera if not given)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--wait-s", type=float, default=600.0, help="start printing once watch/agent connects, or after this")
    sp.add_argument("--linger-s", type=float, default=10.0, help="keep serving after the print ends")

    a = ap.parse_args(argv)
    if a.cmd == "specgen":
        from .parts import get_part
        from .specgen import author_spec, save_spec
        part = get_part(a.part)
        import yaml
        params = {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in a.param)}
        manual = yaml.safe_load(open(a.manual)) if a.manual else None
        method = "heuristic" if a.no_llm else a.method
        res = author_spec(part.layers(), part.layer_h, a.describe or part.spec.get("description", ""),
                          method=method, template=a.template, template_params=params,
                          manual_spec=manual, allow_fallback=not a.strict)
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
        rep = stress_test(layers, part.spec, part.layer_h, a.h, solver=a.solver)
        print(title)
        print(rep.table() if rep.cases else "(no `loads:` in this part's spec; run specgen first)")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(rep.cases):
            if case.fi3 is not None:
                stress_figure(case, title).savefig(out / f"{part.name}_case{i}.png", dpi=140)
        print("verdict:", "PASS: safe to print" if rep.ok else "FAIL: " + "; ".join(rep.violations))
        if a.converge:
            from .fea.verify import convergence, converged
            rows = convergence(layers, part.spec, part.layer_h, solver=a.solver)
            for h, sf in rows:
                print(f"  h={h:4.2f} mm  safety factors: " + ", ".join(f"{v:.2f}" for v in sf))
            print("mesh-converged (finest two within 10%):", converged(rows))
        return
    if a.cmd == "fea-verify":
        from .fea.verify import format_rows, verify
        print(format_rows(verify(a.solver)))
        return
    if a.cmd == "calibrate":
        from .live import calibrate
        calibrate(a.ip, a.code, a.out, a.calib_z, square_mm=a.square_mm, board_origin=tuple(a.board_origin))
        return
    if a.cmd == "connection-test":
        from .connection_test import MockPrinter, RealPrinter, run
        if a.mock:
            printer = MockPrinter(a.mock_broken)
        elif a.ip and a.serial and a.code:
            printer = RealPrinter(a.ip, a.serial, a.code)
        else:
            raise SystemExit("give --ip, --serial and --code (from the printer screen), or --mock")
        res = run(printer, a.out, commands=a.commands)
        raise SystemExit(0 if res["ok"] else 1)
    if a.cmd == "llm-check":
        from .agent.cli import llm_check
        llm_check(a)
        return
    if a.cmd == "agent":
        from .agent.cli import main as agent_main
        agent_main(a)
        return
    if a.cmd == "check-gcode":
        from .gcode import inspect_gcode
        from .parts import Part
        from .slicing import read_gcode_3mf
        rep = inspect_gcode(read_gcode_3mf(a.file, a.plate))
        for k, v in rep.items():
            print(f"{k:22s} {v}")
        try:
            part = Part.from_gcode(a.file, plate=a.plate)
            part.layers()                     # also checks the part fits the camera field of view
        except ValueError as e:
            raise SystemExit(f"NOT USABLE: {e}")
        from .viz import layer_preview
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{part.name}_layers.png"
        layer_preview(part.layers(), f"{part.name}: {len(part.layers())} layers, {part.layer_h} mm").savefig(path, dpi=140)
        ok = rep["layers_with_material"] > 0 and not part.dropped_layers and \
            rep["slicer_total_layers"] in (None, len(part.layers()))
        print(f"preview -> {path}")
        print("verdict:", "looks usable" if ok else "CHECK: layer count/empty layers differ from the slicer's")
        return
    if a.cmd == "probe":
        from .live import probe
        probe(a.ip, a.serial, a.code, a.out)
        return
    if a.cmd == "watch":
        from .live import run_watch
        ports = None
        if a.ports:
            m, c = (int(v) for v in a.ports.split(","))
            ports = dict(mqtt=m, camera=c)
        run_watch(a.part, a.gcode, a.ip, a.serial, a.code, a.calib, a.layer_model, a.out, plate=a.plate,
                  ports=ports, view_port=a.view, view_host=a.view_host, log=lambda *m: print(*m, flush=True))
        return
    if a.cmd == "sim-printer":
        from .printer.fake_p1s import main_sim_printer
        main_sim_printer(a, log=lambda *m: print(*m, flush=True))
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
