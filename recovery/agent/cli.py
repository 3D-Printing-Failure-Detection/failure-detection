"""`python -m recovery.cli agent ...`: build the printer, tools, decider and gate, then run.

Container dent test (the planned pause in Bambu Studio; the person types the dent):
  rehearse:  agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
                   --pause-at-layer 30 --dent 0 -10 3
  real P1S:  agent --part parts/print/water-tray/water-tray.yaml --dent-test \
                   --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
Continuous camera monitoring (top-down camera, calibrated):
  agent --part ... --gcode ... --ip ... --serial ... --code ... --calib calib.npz --monitor on
Decider: --decider rules (default) | local (Qwen on Ollama, or any OpenAI-compatible server) |
         claude (ANTHROPIC_API_KEY) | auto (Claude if a key is set, then local, then rules).
Check a model before a print:  python -m recovery.cli llm-check --decider local
"""
import json
import os
import time

import yaml

from ..parts import Part, get_part
from ..loop.llm import MODEL as CLAUDE_MODEL
from .deciders import LLMDecider, RuleDecider, openai_complete
from .gate import ActionGate, AutoApprover, TerminalApprover
from .loop import run_agent
from .tools import PrinterTools

GEOMETRY_KEYS = ("geometry", "stl", "gcode", "height", "layer_h", "name", "plate", "spec_file")


def add_arguments(p):
    p.add_argument("--part", required=True, help="part YAML: its function spec (and geometry for --sim)")
    p.add_argument("--gcode", help="real print: the Bambu Studio .gcode.3mf being printed")
    p.add_argument("--plate", type=int, default=1)
    p.add_argument("--ip"), p.add_argument("--serial"), p.add_argument("--code")
    p.add_argument("--calib", default="calib.npz")
    p.add_argument("--sim", action="store_true", help="rehearse on the simulated printer")
    p.add_argument("--dent", type=float, nargs=3, metavar=("X", "Y", "DEPTH"),
                   help="sim: hand dent at part-centred (X, Y) mm, DEPTH mm deep")
    p.add_argument("--poke", type=float, nargs=2, metavar=("X", "Y"), help="sim: poke through a wall at (X, Y)")
    p.add_argument("--at-layer", type=int, default=None, help="sim without --dent-test: layer of the dent "
                   "(default: half height)")
    p.add_argument("--dent-test", action="store_true", help="decide at a pause planned in the slicer, "
                   "after a person dents the part")
    p.add_argument("--pause-at-layer", type=int, default=None, help="dent test: the layer the pause is "
                   "before (sim: where to pause; real: must match the pause added in Bambu Studio)")
    p.add_argument("--resume-test", action="store_true", help="at the planned pause, send the ORIGINAL "
                   "remaining layers through stop -> upload -> start (no dent), to prove 'correct' works")
    p.add_argument("--measure", choices=["manual", "camera"], default="manual",
                   help="dent test: the person types the dent (default), or the top-down camera measures it")
    p.add_argument("--monitor", choices=["auto", "on", "off"], default="auto",
                   help="camera check every layer; auto = on in --sim, off on the real P1S (its side-on "
                        "camera does not give the top-down view the measurement needs yet)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--models", default="classical", choices=["classical", "torch"])
    add_decider_arguments(p)
    p.add_argument("--no-auto-pause", action="store_true", help="ask before pausing, keep printing meanwhile")
    p.add_argument("--auto-approve", action="store_true", help="sim only: accept every recommendation")
    p.add_argument("--out", default="results/agent")


def add_decider_arguments(p):
    p.add_argument("--decider", default="rules", choices=["rules", "local", "claude", "auto"])
    p.add_argument("--local-model", default="qwen3:1.7b", help="local model (Ollama tag), e.g. qwen3:1.7b (~1.4 GB) "
                   "or qwen3:8b (~5 GB, better) on 16 GB machines")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434", help="local OpenAI-compatible server")
    p.add_argument("--claude-model", default=CLAUDE_MODEL)


def _claude_complete(model):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(system, user):
        msg = client.messages.create(model=model, max_tokens=400, system=system,
                                     messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    return complete


def make_decider(kind, local_model="qwen3:1.7b", endpoint="http://127.0.0.1:11434", claude_model=CLAUDE_MODEL):
    """rules; local -> rules; claude -> rules; auto: claude (if a key is set) -> local -> rules."""
    if kind == "rules":
        return RuleDecider()
    local = LLMDecider(openai_complete(endpoint, local_model), name=f"local:{local_model}")
    if kind == "local":
        return local
    if kind == "auto" and not os.environ.get("ANTHROPIC_API_KEY"):
        return local
    return LLMDecider(_claude_complete(claude_model), name=f"claude:{claude_model}",
                      fallback=local if kind == "auto" else None)


# a situation like the one the agent sends after a dent next to a hole (the right answer: stop)
SAMPLE = dict(observation=dict(layer=29, z=6.0, label="function_risk", deviation_mm=0.45, paused=True),
              function_if_continue=dict(continue_ok=False, features=[
                  dict(feature="layers supported by the layer below", ok=True, detail=""),
                  dict(feature="hole at [-15, 0] (dia 5.1-6.0 mm)", ok=True, detail=""),
                  dict(feature="hole at [15, 0] (dia 5.1-6.0 mm)", ok=False,
                       detail="hole at [15, 0] diameter 4.7 outside [5.1, 6.0]")]),
              cost=dict(filament_used_mm=929, filament_total_mm=1858, fraction_done=0.5))
SAMPLE_OK = dict(SAMPLE, observation=dict(SAMPLE["observation"], label="under_extrusion", deviation_mm=2.1),
                 function_if_continue=dict(continue_ok=True, features=[
                     dict(f, ok=True, detail="") for f in SAMPLE["function_if_continue"]["features"]]))
OPTIONS = [dict(id="resume", effect="keep printing the original plan"),
           dict(id="stop", effect="stop the print; it is scrapped and must be restarted")]


def llm_check(a):
    """Two canned alarms (one cosmetic, one that breaks a hole) through the chosen decider."""
    dec = make_decider(a.decider, a.local_model, a.endpoint, a.claude_model)
    right = 0
    for name, sit, want in (("cosmetic dent", SAMPLE_OK, "resume"), ("hole narrowed", SAMPLE, "stop")):
        t0 = time.time()
        out = dec.decide(sit, OPTIONS)
        ok = out["action"] == want
        right += ok
        print(f"{name:14s} -> {out['action']:6s} ({'right' if ok else 'WRONG, expected ' + want}) "
              f"by {out['source']} in {time.time() - t0:.1f}s\n    reason: {out['reason']}")
    used = {r.get("decider"): "error: " + r["error"] if "error" in r else "ok"
            for r in getattr(dec, "full_transcript", [])}
    print(f"{right}/2 right; model calls: {json.dumps(used)}")
    return right


def _spec_only(path):
    d = yaml.safe_load(open(path)) or {}
    return {k: v for k, v in d.items() if k not in GEOMETRY_KEYS}


def main(a):
    from ..run import make_models
    from .loop import DentTest
    from .measure import CameraMeasure, ManualMeasure, ScriptedMeasure
    monitor = (a.monitor == "on") or (a.monitor == "auto" and a.sim)
    camera_needed = monitor or (a.dent_test and a.measure == "camera")
    seg, det = make_models(a.models) if camera_needed else (None, None)
    change = (f"dent {a.dent[0]} {a.dent[1]} {a.dent[2]}" if a.dent else
              f"poke {a.poke[0]} {a.poke[1]}" if a.poke else "none")
    if a.sim:
        from .rehearse import PlannedPauseSim, hand_dent, hand_poke
        part = get_part(a.part)
        nom = part.layers()
        if a.dent_test or a.resume_test:
            k = a.pause_at_layer if a.pause_at_layer is not None else len(nom) // 2
            printer = PlannedPauseSim(nom, None, seed=a.seed, pause_at=k)
            measure = CameraMeasure() if a.measure == "camera" else ScriptedMeasure(change)
            test = DentTest(person=lambda: printer.user_change(change), measure=measure)
        else:
            k = a.at_layer if a.at_layer is not None else len(nom) // 2 - 1
            failure = (hand_dent(nom, *a.dent, k) if a.dent else hand_poke(nom, *a.poke, k) if a.poke else None)
            from ..sim import SimPrinter
            printer, test = SimPrinter(nom, failure, seed=a.seed), None
    else:
        if a.auto_approve:
            raise SystemExit("--auto-approve is for --sim only: a person approves actions on a real print")
        if not (a.gcode and a.ip and a.serial and a.code):
            raise SystemExit("a real run needs --gcode, --ip, --serial and --code (or use --sim)")
        if camera_needed and not os.path.exists(a.calib):
            raise SystemExit(f"camera measurement needs a calibration ({a.calib}); see `calibrate`, "
                             "or use --dent-test --measure manual --monitor off")
        from ..camera import BedCamera
        from ..printer.bambu import BambuPrinter
        part = Part.from_gcode(a.gcode, spec=_spec_only(a.part), plate=a.plate)
        nom = part.layers()
        printer = BambuPrinter(a.ip, a.serial, a.code, nom, part.layer_h,
                               BedCamera.load(a.calib) if camera_needed else None,
                               origin=part.origin, original_gcode=part.gcode_text)
        test = None
        printer.notify_pause = bool(a.dent_test or a.resume_test)
        if a.dent_test:
            where = f"layer {a.pause_at_layer}" if a.pause_at_layer is not None else "the layer you chose"
            test = DentTest(person=lambda: input(f"\nPrinter paused at {where}. Make the dent or poke now "
                                                 "(tool, not fingers), then press Enter... "),
                            measure=CameraMeasure() if a.measure == "camera" else ManualMeasure())
    tools = PrinterTools(printer, part, seg, det)
    approver = AutoApprover() if a.auto_approve else TerminalApprover()
    gate = ActionGate(tools, approver, auto_pause=not a.no_auto_pause)
    try:
        return run_agent(tools, make_decider(a.decider, a.local_model, a.endpoint, a.claude_model), gate,
                         out_dir=a.out, monitor=monitor, dent_test=test, resume_test=a.resume_test)
    finally:
        if hasattr(printer, "close"):
            printer.close()
