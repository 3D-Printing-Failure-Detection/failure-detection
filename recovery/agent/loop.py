"""The agent loop: every layer, look; on an alarm, pause, ask the decider, get approval, act, log.

Two kinds of alarm: the outline is more than ~1 mm off plan (detector), or the measured layer
breaks a functional feature even though it looks almost right (a hole narrowed by a 1 mm dent
moves the outline by less than the detector's threshold). The second is checked on every layer.

Resume test (`resume_test=True`): at the planned pause, with no change to the part, send the
ORIGINAL remaining layers through the correction path (stop -> upload -> start). If that tray
finishes cleanly, the mechanism behind "correct" works on this printer.

Dent test: a person pauses the printer (screen, app, or a pause planned in Bambu Studio). The
agent notices the pause, saves a camera frame, the person makes the change (dent or poke) and
presses Enter, the agent saves another frame, gets the measurement (typed in, or from the camera),
checks function and decides while the printer is still paused: resume (still works), correct (new
remaining layers that return to the original shape), or stop. The person must not resume on the
printer; if they do anyway, the agent offers "continue" instead of "resume" and sends nothing.
Until the pause, the agent only reads the printer's status: no camera, no decider calls.

An acknowledged deviation (we resumed and the part still worked) does not raise a new alarm on
every following layer; only a function check that now fails, or a deviation that has grown by
more than `regrow_mm`, does. That keeps a person from being asked the same question 30 times.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..printer.bambu import PrintEnded


@dataclass
class DentTest:
    person: Callable[[], None]       # makes the change (sim) or asks the person to (real printer)
    measure: Callable               # tools -> (new top-layer shapes or None, description)


def _summary(situation, rec):
    o, c = situation["observation"], situation["cost"]
    lines = [f"\n=== {o['label'].upper()} at layer {o['layer']} (z = {o['z']:.2f} mm)"
             + (f": {o['deviation_mm']:.2f} mm off plan" if o.get("deviation_mm") is not None else "")
             + (f"; {o['described']}" if o.get("described") else ""), "If we continue, the finished part:"]
    for f in situation["function_if_continue"]["features"]:
        lines.append(f"  {'OK  ' if f['ok'] else 'FAIL'} {f['feature']}" + (f"  ({f['detail']})" if f["detail"] else ""))
    fix = situation.get("correction")
    if fix:
        back = f"back to the original over {fix['fade_layers']} layers" if fix["fade_layers"] else "keeping the new shape"
        lines.append(f"Correction available: new remaining layers, {back}; every feature passes.")
        lines.append(f"  Its G-code and a preview are saved; inspect before approving: {fix['gcode']}, {fix['preview']}")
    lines.append(f"Filament: {c['filament_used_mm']} mm used of {c['filament_total_mm']} "
                 f"({c['fraction_done']:.0%} done)")
    lines.append(f"Agent ({rec['source']}): {rec['action'].upper()} because {rec['reason']}")
    return "\n".join(lines)


def _frame_change(before, after):
    """How much of the camera view changed between two frames of the paused printer."""
    a, b = (cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (5, 5), 0) for x in (before, after))
    mask = (cv2.absdiff(a, b) > 25).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    return mask * 255, dict(changed_fraction=round(float(mask.mean()), 5), changed_box_px=box)


class _Run:
    def __init__(self, tools, decider, gate, out_dir, log):
        self.tools, self.decider, self.gate, self.log = tools, decider, gate, log
        self.dir = Path(out_dir) / time.strftime("%Y%m%d-%H%M%S")
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.events = []
        self.file = open(self.dir / "events.jsonl", "w")

    def save(self, name, image):
        cv2.imwrite(str(self.dir / "frames" / name), image)
        return f"frames/{name}"

    def save_plan(self, layers, label):
        """The replacement G-code and a picture of its first layers, before anyone approves it."""
        from ..viz import layer_preview
        k = self.tools.pr.n_done
        gcode = self.dir / f"{label}_from_layer_{k}.gcode"
        gcode.write_text(self.tools.correction_gcode(layers))
        shown = [self.tools.known[-1]] + list(layers[:12])
        fig = layer_preview(shown, f"{label}: printed top layer, then the next {len(shown) - 1} new layers", n=4)
        preview = self.dir / f"{label}_from_layer_{k}.png"
        fig.savefig(preview, dpi=120)
        return str(gcode), str(preview)

    def decide(self, observation, report, paused, frames, extra=None):
        tools = self.tools
        options = [dict(id="resume" if paused else "continue", effect="keep printing the original plan")]
        situation = dict(observation=observation, function_if_continue=report, cost=tools.cost(),
                         status=tools.status(), **(extra or {}))
        fix = None
        if not report["continue_ok"]:
            fix = tools.correction()
            if fix:
                n = tools.pr.n_total - tools.pr.n_done
                back = (f"return to the original shape over {fix['fade_layers']} layers"
                        if fix["fade_layers"] else "keep the new shape to the top")
                options.append(dict(id="correct", effect=f"replace the remaining {n} layers: start on the "
                                    f"measured shape and {back}; every feature passes"))
                gcode, preview = self.save_plan(fix["layers"], "correction")
                situation["correction"] = dict(fade_layers=fix["fade_layers"], fidelity=fix["fidelity"],
                                               features=fix["report"]["features"], gcode=gcode, preview=preview)
        options.append(dict(id="stop", effect="stop the print; it is scrapped and must be restarted"))
        rec = self.decider.decide(situation, options)
        action = self.gate.decide_and_act(rec, options, _summary(situation, rec),
                                          correction_layers=fix["layers"] if fix else None)
        event = dict(t=time.time(), situation=situation, options=[o["id"] for o in options],
                     recommendation=rec, action=action, frames=frames)
        self.file.write(json.dumps(event, default=str) + "\n")
        self.file.flush()
        self.events.append(event)
        self.log(f"layer {observation['layer']}: {observation['label']} -> {action} ({rec['source']})")
        return action

    def dent_test(self, test: DentTest):
        tools = self.tools
        layer = tools.pr.n_done - 1
        before = tools.snapshot()
        frames = dict(before=self.save(f"{layer:04d}_before.png", before))
        test.person()
        after = tools.snapshot()
        frames["after"] = self.save(f"{layer:04d}_after.png", after)
        diff, change = _frame_change(before, after)
        frames["change"] = self.save(f"{layer:04d}_change.png", diff)
        tools.pause_frames = (before, after)          # for a measurement that compares the two frames
        new, described = test.measure(tools)
        if new:
            tools.believe_top(new)
        paused = tools.is_paused()
        if not paused:
            self.log("the printer was resumed while the agent was waiting: new layers are printing "
                     "over the change; deciding anyway")
        observation = dict(layer=layer, z=tools.known[-1].z, label="dent test", described=described,
                           camera_change=change, paused=paused)
        return self.decide(observation, tools.function_report(), paused=paused, frames=frames)

    def resume_test(self):
        """Send the unchanged remaining layers through stop -> upload -> start, after approval."""
        tools = self.tools
        layer = tools.pr.n_done - 1
        frames = dict(before=self.save(f"{layer:04d}_before.png", tools.snapshot()))
        layers = tools.pr.plan[tools.pr.n_done:]
        gcode, preview = self.save_plan(layers, "resume_test")
        options = [dict(id="restart", effect="stop the job, upload the ORIGINAL remaining layers as a new "
                        "file and start it (tests the mechanism behind 'correct')"),
                   dict(id="resume", effect="normal resume instead (no test)"),
                   dict(id="stop", effect="stop the print")]
        rec = dict(action="restart", reason="resume test requested", source="test")
        summary = (f"\n=== RESUME TEST at layer {layer}: the part is unchanged.\n"
                   f"The printer will stop, then start {gcode} ({len(layers)} layers).\n"
                   f"Watch the bed: if it homes Z or moves toward the nozzle over the part, stop the printer.")
        action = self.gate.decide_and_act(rec, options, summary, correction_layers=layers)
        event = dict(t=time.time(), situation=dict(observation=dict(layer=layer, label="resume test"),
                     gcode=gcode, preview=preview), options=[o["id"] for o in options],
                     recommendation=rec, action=action, frames=frames)
        self.file.write(json.dumps(event, default=str) + "\n")
        self.file.flush()
        self.events.append(event)
        self.log(f"layer {layer}: resume test -> {action}")
        return action

    def close(self):
        self.file.close()


def run_agent(tools, decider, gate, out_dir="results/agent", regrow_mm=0.5, monitor=True,
              dent_test: DentTest = None, resume_test=False, log=print):
    r = _Run(tools, decider, gate, out_dir, log)
    acknowledged, outcome = None, "finished"
    log(f"agent watching {tools.pr.n_total} layers (camera monitoring {'on' if monitor else 'off'}"
        f"{', dent test at the planned pause' if dent_test else ''}); log -> {r.dir}")
    try:
        while tools.pr.n_done < tools.pr.n_total and not tools.pr.cancelled:
            if tools.next_layer() == "paused":
                if resume_test:
                    action = r.resume_test()
                elif dent_test is None:
                    raise RuntimeError("the printer paused for a dent test, but no dent test was set up")
                else:
                    action = r.dent_test(dent_test)
                if action == "stop":
                    outcome = "stopped"
                    break
                acknowledged = float("inf") if action == "resume" else acknowledged
                continue
            if not monitor:
                continue
            obs = tools.observe()
            if obs.label == "ok":
                report = tools.function_report(current=obs.measured)
                if report["continue_ok"]:
                    continue
                obs.label = "function_risk"                      # looks close to plan, but breaks a feature
                tools.believe_measured(obs)
            else:
                report = tools.function_report()
            if acknowledged and report["continue_ok"] and obs.deviation_mm <= acknowledged + regrow_mm:
                continue                                         # same, already-accepted deviation
            paused = gate.pause_for_alarm()
            observation = dict(layer=obs.layer, z=obs.z, label=obs.label,
                               deviation_mm=round(obs.deviation_mm, 2), paused=paused)
            action = r.decide(observation, report, paused, frames=dict(alarm=r.save(f"{obs.layer:04d}.png", obs.image)))
            if action == "stop":
                outcome = "stopped"
                break
            acknowledged = max(acknowledged or 0.0, obs.deviation_mm)
    except PrintEnded as e:
        outcome = f"ended on the printer: {e}"
    except KeyboardInterrupt:
        outcome = "agent interrupted (the print itself is not affected)"
    finally:
        r.close()
    transcript = getattr(decider, "full_transcript", None)
    if transcript:
        (r.dir / "llm_transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    summary = dict(outcome=outcome, layers=tools.pr.n_done, decisions=len(r.events), decider=decider.name,
                   decided_by=[e["recommendation"]["source"] for e in r.events],
                   actions=[e["action"] for e in r.events])
    (r.dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"{outcome} after {tools.pr.n_done} layers, {len(r.events)} decision(s); see {r.dir}")
    return dict(summary, run=str(r.dir), events=r.events)
