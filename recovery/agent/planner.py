"""Who designs the correction when continuing as planned would break the part.

  search (default with --decider rules): try fade lengths 10, 15, 30 and "keep the new shape", keep
         the passing plan closest to the original design (PrinterTools.correction).
  llm    a language model proposes the plan (how many layers to return to the original over, or keep
         the new shape) with a reason; our code builds it (Ishaan's contour fit) and checks it with
         the same function checks (Sri's spec). A failing plan goes back to the model with the
         reasons, up to `rounds` times. If no proposal passes, or the model is unavailable, the
         search runs instead, and the log says so.

With `writer` (an LLMGcodeWriter, `--plan-fix gcode`) the model then also writes the G-code of the
changed layers (recovery/agent/gcode_writer.py); otherwise our code writes it. Nothing is offered
unless every check passes.
"""
import json
import time

from .deciders import parse_reply
from .usage import labelled

SYSTEM = """You plan a repair for a 3D print that was damaged mid-print (e.g. a wall dented inward).
The layers already printed cannot change. You choose how the REMAINING layers should be shaped:
  - they always start exactly on the measured (damaged) top layer, so the first new layer is supported;
  - then they return to the original design over `fade_layers` layers (a straight blend), or,
    with fade_layers = null, keep the damaged shape all the way to the top.
Trade-off: a short fade returns to the design sooner (the part looks and fits more like the
original), but each new layer then steps further sideways over the one below and can hang over air
(each layer may shift by about depth / fade_layers mm). A long fade, or null, is gentler but leaves
the part further from the design. Every proposal is built and checked against the part's function
(e.g. holds liquid, holes in tolerance, every new layer supported); you see the result of earlier
proposals in `tried`. Do not repeat a failed proposal; use its failure reasons.
Reply with JSON only: {"fade_layers": <a value allowed by fade_layers_allowed>,
"reason": "<one or two sentences>"}"""


class SearchPlanner:
    name = "search"

    def plan(self, tools, situation):
        fix = tools.correction()
        if fix:
            fix.update(planned_by=self.name, attempts=[])
        return fix


def _summary(fix):
    return dict(fade_layers=fix["fade_layers"], passes=fix["report"]["continue_ok"], fidelity=fix["fidelity"],
                failing=[f"{f['feature']}: {f['detail']}" for f in fix["report"]["features"] if not f["ok"]])


class LLMFixPlanner:
    """A language model proposes correction plans; the function checks accept or reject each one."""

    def __init__(self, complete, name="llm", rounds=3, fallback=None, system=SYSTEM, writer=None):
        self.complete, self.name, self.rounds, self.system = labelled(complete, "plan"), name, rounds, system
        self.fallback = fallback or SearchPlanner()
        self.writer = writer
        self.transcript = []

    def plan(self, tools, situation):
        fix = self._plan(tools, situation)
        if fix and self.writer is not None:
            fix = self.writer.write(tools, fix)
        return fix

    def _plan(self, tools, situation):
        o = situation["observation"]
        remaining = tools.pr.n_total - tools.pr.n_done
        # when the model also writes the G-code, every changed layer is a model call: cap them
        limit = min(remaining, self.writer.max_layers) if self.writer is not None else remaining
        keep_ok = self.writer is None
        allowed = f"an integer from 1 to {limit}" + (", or null (keep the damaged shape)" if keep_ok else "")
        base = dict(measured_change=o.get("described") or o.get("label"), after_layer=o["layer"] + 1,
                    layer_height_mm=tools.part.layer_h, remaining_layers=remaining, fade_layers_allowed=allowed,
                    part=tools.part.spec.get("description", ""),
                    if_we_continue_unchanged=[f"{f['feature']}: {f['detail']}"
                                              for f in situation["function_if_continue"]["features"] if not f["ok"]])
        tried, why = [], "no proposal passed"
        for _ in range(self.rounds):
            user = json.dumps(dict(base, tried=tried), indent=1, default=str)
            rec, t0 = dict(planner=self.name, prompt=user), time.time()
            try:
                reply = self.complete(self.system, user)
                rec["reply"] = reply
                ans = parse_reply(reply)
                fade = ans.get("fade_layers")
                fade = None if fade is None else int(fade)
            except Exception as e:           # no key, server down, bad reply -> the search, visibly
                rec.update(error=f"{type(e).__name__}: {e}"[:300], latency_s=round(time.time() - t0, 2))
                self.transcript.append(rec)
                why = f"{self.name} unavailable or invalid: {type(e).__name__}"
                break
            if (fade is None and not keep_ok) or (fade is not None and not 1 <= fade <= limit):
                fix, result = None, dict(fade_layers=fade, passes=False, failing=[f"not allowed: use {allowed}"])
            else:
                fix = tools.try_correction(fade)
                result = _summary(fix) if fix else dict(fade_layers=fade, passes=False, failing=["nothing left to print"])
            tried.append(dict(result, reason=str(ans.get("reason", ""))))
            rec.update(latency_s=round(time.time() - t0, 2), result=result)
            self.transcript.append(rec)
            if fix and fix["report"]["continue_ok"]:
                fix.update(planned_by=self.name, reason=str(ans.get("reason", "")), attempts=tried)
                return fix
        fix = self.fallback.plan(tools, situation)
        if fix:
            fix.update(planned_by=f"{self.name}->{fix['planned_by']}", reason=why, attempts=tried)
        return fix

    @property
    def full_transcript(self):
        return self.transcript + (self.writer.full_transcript if self.writer is not None else [])
