"""`python -m recovery.cli agent ...`: build the printer, tools, decider and gate, then run.

Everything the agent sees comes from the printer's own camera (front-left, low, wide lens), in sim
and on the real P1S alike: there is no top-down view.

Container dent test (a pause planned in Bambu Studio; the camera tells dent / poke / none):
  rehearse:  agent --part parts/print/water-tray/water-tray.yaml --sim --dent-test \
                   --pause-at-layer 30 --dent 0 -10 3
  real P1S:  agent --part parts/print/water-tray/water-tray.yaml --dent-test \
                   --gcode water-tray.gcode.3mf --ip IP --serial SERIAL --code CODE
Watching every layer for failures (on whenever the trained model is there):
  agent --part ... --gcode ... --ip ... --serial ... --code ... [--calib calib.npz]
Decider: --decider rules (default) | openai (OPENAI_API_KEY, --openai-model) |
         local (Qwen on Ollama, or any OpenAI-compatible server) | claude (ANTHROPIC_API_KEY) |
         auto (Claude if its key is set, then OpenAI if its key is set, then local, then rules).
         Keys can sit in a .env file in the folder you run from (KEY=value lines; never commit it).
Check a model before a print:  python -m recovery.cli llm-check --decider openai
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

OPENAI_MODEL = "gpt-4.1-mini"
GEOMETRY_KEYS = ("geometry", "stl", "gcode", "height", "layer_h", "name", "plate", "spec_file")


def add_arguments(p):
    p.add_argument("--part", required=True, help="part YAML: its function spec (and geometry for --sim)")
    p.add_argument("--gcode", help="real print: the Bambu Studio .gcode.3mf being printed")
    p.add_argument("--plate", type=int, default=1)
    p.add_argument("--ip"), p.add_argument("--serial"), p.add_argument("--code")
    p.add_argument("--ports", help="MQTT,camera ports if not 8883,6000 (the simulated printer: recovery.cli sim-printer)")
    p.add_argument("--calib", default="calib.npz")
    p.add_argument("--sim", action="store_true", help="rehearse on the simulated printer")
    p.add_argument("--dent", type=float, nargs=3, metavar=("X", "Y", "DEPTH"),
                   help="sim: hand dent at part-centred (X, Y) mm, DEPTH mm deep")
    p.add_argument("--poke", type=float, nargs=2, metavar=("X", "Y"), help="sim: poke through a wall at (X, Y)")
    p.add_argument("--at-layer", type=int, default=None, help="sim without --dent-test: layer of the dent "
                   "(default: half height)")
    p.add_argument("--dent-test", action="store_true", help="when a person pauses the printer (screen, "
                   "app or a pause planned in Bambu Studio) and dents the part, decide what to do")
    p.add_argument("--pause-at-layer", type=int, default=None, help="--sim only: pause after this many "
                   "layers (on the real printer the agent reacts to whatever pause the person makes)")
    p.add_argument("--resume-test", action="store_true", help="at the pause, send the ORIGINAL "
                   "remaining layers through stop -> upload -> start (no dent), to prove 'correct' works")
    p.add_argument("--measure", choices=["sideview", "manual"], default=None,
                   help="dent test: the person types the change (manual; the default on the real printer), or "
                        "the camera compares the frames before and after it and the person types only a dent's "
                        "depth (sideview; the default with --sim, opt-in on the real printer until it is scored "
                        "on real dent-test photos). Falls back to manual if --dent-model is missing")
    p.add_argument("--dent-model", default="runs/dent", help="dent detector checkpoint (recovery/models/dent.py)")
    p.add_argument("--layer-model", default="runs/p1s_layer",
                   help="failure detector checkpoint for the every-layer check (recovery/models/p1s_layer.py)")
    p.add_argument("--monitor", choices=["auto", "on", "off"], default="auto",
                   help="camera check every layer; auto = on with --sim when --layer-model exists, OFF on the "
                        "real printer (each alarm pauses a real print and the false-alarm rate on real frames "
                        "is not measured yet: use 'on' to opt in)")
    p.add_argument("--seed", type=int, default=1)
    add_decider_arguments(p)
    p.add_argument("--no-auto-pause", action="store_true", help="ask before pausing, keep printing meanwhile")
    p.add_argument("--auto-approve", action="store_true", help="sim only: accept every recommendation")
    p.add_argument("--out", default="results/agent")


def add_decider_arguments(p):
    p.add_argument("--decider", default="rules", choices=["rules", "openai", "local", "claude", "auto"])
    p.add_argument("--openai-model", default=OPENAI_MODEL, help="OpenAI model for --decider openai/auto, "
                   "e.g. gpt-4.1-mini (default), gpt-4o-mini, gpt-5.4-mini")
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


def load_dotenv(path=".env"):
    """KEY=value lines from a .env file into the environment (already-set variables win)."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _openai(model, fallback=None):
    return LLMDecider(openai_complete("https://api.openai.com", model, api_key=os.environ["OPENAI_API_KEY"]),
                      name=f"openai:{model}", fallback=fallback)


def make_decider(kind, local_model="qwen3:1.7b", endpoint="http://127.0.0.1:11434", claude_model=CLAUDE_MODEL,
                 openai_model=OPENAI_MODEL):
    """rules; openai -> rules; local -> rules; claude -> rules;
    auto: claude (if its key is set) -> openai (if its key is set) -> local -> rules."""
    load_dotenv()
    if kind == "rules":
        return RuleDecider()
    if kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--decider openai needs OPENAI_API_KEY (in the environment or a .env file here)")
        return _openai(openai_model)
    local = LLMDecider(openai_complete(endpoint, local_model), name=f"local:{local_model}")
    if kind == "local":
        return local
    if kind == "auto":
        chain = _openai(openai_model, fallback=local) if os.environ.get("OPENAI_API_KEY") else local
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return chain
        return LLMDecider(_claude_complete(claude_model), name=f"claude:{claude_model}", fallback=chain)
    return LLMDecider(_claude_complete(claude_model), name=f"claude:{claude_model}")


# a situation like the one the agent sends after a dent next to a hole (the right answer: stop)
SAMPLE = dict(observation=dict(layer=29, z=6.0, label="dent test", described="dent 13 0 1", paused=True),
              function_if_continue=dict(continue_ok=False, features=[
                  dict(feature="layers supported by the layer below", ok=True, detail=""),
                  dict(feature="hole at [-15, 0] (dia 5.1-6.0 mm)", ok=True, detail=""),
                  dict(feature="hole at [15, 0] (dia 5.1-6.0 mm)", ok=False,
                       detail="hole at [15, 0] diameter 4.7 outside [5.1, 6.0]")]),
              cost=dict(filament_used_mm=929, filament_total_mm=1858, fraction_done=0.5))
SAMPLE_OK = dict(SAMPLE, observation=dict(layer=29, z=6.0, label="under_extrusion", flagged_px=120, grown=False,
                                         paused=True),
                 function_if_continue=dict(continue_ok=True, features=[
                     dict(f, ok=True, detail="") for f in SAMPLE["function_if_continue"]["features"]]))
OPTIONS = [dict(id="resume", effect="keep printing the original plan"),
           dict(id="stop", effect="stop the print; it is scrapped and must be restarted")]


def llm_check(a):
    """Two canned alarms (one cosmetic, one that breaks a hole) through the chosen decider."""
    dec = make_decider(a.decider, a.local_model, a.endpoint, a.claude_model, a.openai_model)
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


def _camera(a):
    """The P1S camera: the calibration if there is one that records the bed Z (calibrate-bed /
    calibrate-board --calib-z), otherwise the approximate default camera."""
    from .. import p1s_view as V
    if os.path.exists(a.calib):
        try:
            return V.P1SCamera.load(a.calib)
        except ValueError as e:
            print(f"{e}; using the approximate default P1S camera")
    else:
        print(f"no calibration at {a.calib}: using the approximate default P1S camera (recovery/p1s_view.py)")
    return None


def _has_model(path):
    return os.path.exists(os.path.join(path, "model.pt"))


def resolve_modes(a):
    """Sets a.measure and returns whether to watch every layer. On the real printer both camera checks are
    opt-in (--measure sideview, --monitor on) until they are scored on real frames; --sim uses them."""
    if a.measure is None:
        a.measure = "sideview" if a.sim else "manual"
    return a.monitor == "on" or (a.monitor == "auto" and a.sim and _has_model(a.layer_model))


def main(a):
    from .loop import DentTest
    from .measure import ManualMeasure, ScriptedMeasure, SideViewMeasure
    cam = _camera(a)
    monitor = resolve_modes(a)
    if monitor and not _has_model(a.layer_model):
        raise SystemExit(f"no failure detector at {a.layer_model}: train one (python -m recovery.models.p1s_layer "
                         "train ...) or use --monitor off")
    if a.monitor == "auto" and not monitor:
        print(f"no failure detector at {a.layer_model}: not watching every layer" if a.sim else
              "real printer: not watching every layer (--monitor on to opt in)")
    if a.dent_test and a.measure == "sideview" and not _has_model(a.dent_model):
        print(f"no dent detector at {a.dent_model} (train one: python -m recovery.models.dent train ...); "
              "the person types the change instead")
        a.measure = "manual"
    watcher = None
    if monitor:
        from ..models.p1s_layer import LayerWatcher
        watcher = LayerWatcher(a.layer_model, cam)
    change = (f"dent {a.dent[0]} {a.dent[1]} {a.dent[2]}" if a.dent else
              f"poke {a.poke[0]} {a.poke[1]}" if a.poke else "none")
    if a.sim:
        from .rehearse import PlannedPauseSim, hand_dent, hand_poke
        part = get_part(a.part)
        nom = part.layers()
        if a.dent_test or a.resume_test:
            k = a.pause_at_layer if a.pause_at_layer is not None else len(nom) // 2
            printer = PlannedPauseSim(nom, None, seed=a.seed, pause_at=k, p1s_camera=cam)
            depth = a.dent[2] if a.dent else 1.5          # what the person would type in a rehearsal
            measure = (SideViewMeasure(a.dent_model, cam, depth=lambda *_: depth) if a.measure == "sideview"
                       else ScriptedMeasure(change))
            test = DentTest(person=lambda: printer.user_change(change), measure=measure)
        else:
            k = a.at_layer if a.at_layer is not None else len(nom) // 2 - 1
            failure = (hand_dent(nom, *a.dent, k) if a.dent else hand_poke(nom, *a.poke, k) if a.poke else None)
            printer, test = PlannedPauseSim(nom, failure, seed=a.seed, p1s_camera=cam), None
    else:
        if a.auto_approve:
            raise SystemExit("--auto-approve is for --sim only: a person approves actions on a real print")
        if not (a.ip and a.serial and a.code):
            raise SystemExit("a real run needs --gcode, --ip, --serial and --code (or use --sim)")
        from ..printer.bambu import BambuPrinter
        if a.gcode:
            part = Part.from_gcode(a.gcode, spec=_spec_only(a.part), plate=a.plate)
        elif a.ports:                                   # the simulated printer prints the YAML shape itself
            part = get_part(a.part)
        else:
            raise SystemExit("a real print needs --gcode (the sliced file being printed)")
        nom = part.layers()
        ports = None
        if a.ports:
            m, c = (int(v) for v in a.ports.split(","))
            ports = dict(mqtt=m, camera=c)
        printer = BambuPrinter(a.ip, a.serial, a.code, nom, part.layer_h,
                               origin=part.origin, original_gcode=part.gcode_text, ports=ports)
        test = None
        printer.notify_pause = bool(a.dent_test or a.resume_test)
        printer.skip_ahead = printer.notify_pause and not monitor     # no per-layer checks: may start mid-print
        if not printer.link.wait_status(30):
            printer.close()
            raise SystemExit("no status from the printer in 30 s: check --ip, --serial and --code (connection-test), "
                             "and close Bambu Studio / Bambu Handy on other devices if they are connected over LAN")
        if printer.notify_pause:
            print(f"Connected. Pause from the printer screen at about half height (layer {len(nom) // 2}, "
                  f"z = {nom[len(nom) // 2 - 1].z:.1f} mm); the agent shows the layer below and then takes over.")
        if a.dent_test:
            test = DentTest(person=lambda: input(f"\nPrinter paused after layer {printer.n_done} of "
                                                 f"{printer.n_total}. Make the dent or poke now "
                                                 "(tool, not fingers), then press Enter here. Do NOT "
                                                 "resume on the printer; the agent does that... "),
                            measure=SideViewMeasure(a.dent_model, cam) if a.measure == "sideview" else ManualMeasure())
    tools = PrinterTools(printer, part, watcher)
    approver = AutoApprover() if a.auto_approve else TerminalApprover()
    gate = ActionGate(tools, approver, auto_pause=not a.no_auto_pause)
    try:
        return run_agent(tools, make_decider(a.decider, a.local_model, a.endpoint, a.claude_model, a.openai_model), gate,
                         out_dir=a.out, monitor=monitor, dent_test=test, resume_test=a.resume_test)
    finally:
        if hasattr(printer, "close"):
            printer.close()
