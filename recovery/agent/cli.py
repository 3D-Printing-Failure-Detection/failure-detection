"""`python -m recovery.cli agent ...`: build the printer, tools, decider and gate, then run.

  rehearse (no printer):  agent --part parts/print/two-hole-block/two-hole-block.yaml --sim \
                                --dent 15 -5 1.0 --at-layer 29 --auto-approve
  real P1S:               agent --part parts/print/two-hole-block/two-hole-block.yaml \
                                --gcode two-hole-block.gcode.3mf --ip IP --serial SERIAL --code CODE \
                                --calib calib.npz
Decider: --decider rules (default) | claude (needs ANTHROPIC_API_KEY, --model) |
         openai (any OpenAI-compatible server, e.g. llama.cpp/Ollama: --endpoint, --model).
"""
import json
import os
import urllib.request

import yaml

from ..parts import Part, get_part
from .deciders import LLMDecider, RuleDecider
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
    p.add_argument("--at-layer", type=int, default=None, help="sim: layer of the dent (default: half height)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--models", default="classical", choices=["classical", "torch"])
    p.add_argument("--decider", default="rules", choices=["rules", "claude", "openai"])
    p.add_argument("--model", default=None, help="model id for --decider claude/openai")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080", help="--decider openai: server URL")
    p.add_argument("--no-auto-pause", action="store_true", help="ask before pausing, keep printing meanwhile")
    p.add_argument("--auto-approve", action="store_true", help="sim only: accept every recommendation")
    p.add_argument("--out", default="results/agent")


def make_decider(kind, model=None, endpoint=None):
    if kind == "rules":
        return RuleDecider()
    if kind == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if not model:
            raise SystemExit("--decider claude needs --model (a Claude model id)")

        def complete(system, user):
            msg = client.messages.create(model=model, max_tokens=400, system=system,
                                         messages=[{"role": "user", "content": user}])
            return msg.content[0].text
        return LLMDecider(complete)

    def complete(system, user):                    # OpenAI-compatible chat completions
        body = dict(model=model or "local", temperature=0,
                    messages=[dict(role="system", content=system), dict(role="user", content=user)])
        req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return LLMDecider(complete)


def _spec_only(path):
    d = yaml.safe_load(open(path)) or {}
    return {k: v for k, v in d.items() if k not in GEOMETRY_KEYS}


def main(a):
    from ..run import make_models
    seg, det = make_models(a.models)
    if a.sim:
        from ..sim import SimPrinter
        from .rehearse import hand_dent, hand_poke
        part = get_part(a.part)
        nom = part.layers()
        k = a.at_layer if a.at_layer is not None else len(nom) // 2 - 1
        failure = (hand_dent(nom, *a.dent, k) if a.dent else hand_poke(nom, *a.poke, k) if a.poke else None)
        printer = SimPrinter(nom, failure, seed=a.seed)
    else:
        if a.auto_approve:
            raise SystemExit("--auto-approve is for --sim only: a person approves actions on a real print")
        if not (a.gcode and a.ip and a.serial and a.code):
            raise SystemExit("a real run needs --gcode, --ip, --serial and --code (or use --sim)")
        if not os.path.exists(a.calib):
            raise SystemExit(f"a real run needs a camera calibration ({a.calib}); see `calibrate`")
        from ..camera import BedCamera
        from ..printer.bambu import BambuPrinter
        part = Part.from_gcode(a.gcode, spec=_spec_only(a.part), plate=a.plate)
        nom = part.layers()
        printer = BambuPrinter(a.ip, a.serial, a.code, nom, part.layer_h, BedCamera.load(a.calib),
                               origin=part.origin, original_gcode=part.gcode_text)
    tools = PrinterTools(printer, part, seg, det)
    approver = AutoApprover() if a.auto_approve else TerminalApprover()
    gate = ActionGate(tools, approver, auto_pause=not a.no_auto_pause)
    try:
        return run_agent(tools, make_decider(a.decider, a.model, a.endpoint), gate, out_dir=a.out)
    finally:
        if hasattr(printer, "close"):
            printer.close()
