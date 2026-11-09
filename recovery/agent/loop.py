"""The agent loop: every layer, look; on an alarm, pause, ask the decider, get approval, act, log.

Two kinds of alarm: the outline is more than ~1 mm off plan (detector), or the measured layer
breaks a functional feature even though it looks almost right (a hole narrowed by a 1 mm dent
moves the outline by less than the detector's threshold). The second is checked on every layer.

An acknowledged deviation (we resumed and the part still worked) does not raise a new alarm on
every following layer; only a function check that now fails, or a deviation that has grown by
more than `regrow_mm`, does. That keeps a person from being asked the same question 30 times.
"""
import json
import time
from pathlib import Path

import cv2

from ..printer.bambu import PrintEnded


def _summary(situation, rec):
    o, c = situation["observation"], situation["cost"]
    lines = [f"\n=== ALARM at layer {o['layer']} (z = {o['z']:.2f} mm): {o['label']}, "
             f"{o['deviation_mm']:.2f} mm off plan", "If we continue, the finished part:"]
    for f in situation["function_if_continue"]["features"]:
        lines.append(f"  {'OK  ' if f['ok'] else 'FAIL'} {f['feature']}" + (f"  ({f['detail']})" if f["detail"] else ""))
    lines.append(f"Filament: {c['filament_used_mm']} mm used of {c['filament_total_mm']} "
                 f"({c['fraction_done']:.0%} done)")
    lines.append(f"Agent ({rec['source']}): {rec['action'].upper()} because {rec['reason']}")
    return "\n".join(lines)


def run_agent(tools, decider, gate, out_dir="results/agent", regrow_mm=0.5, log=print):
    run = Path(out_dir) / time.strftime("%Y%m%d-%H%M%S")
    (run / "frames").mkdir(parents=True, exist_ok=True)
    events, acknowledged, outcome = [], None, "finished"
    log(f"agent watching {tools.pr.n_total} layers; log -> {run}")
    with open(run / "events.jsonl", "w") as f:
        try:
            while tools.pr.n_done < tools.pr.n_total and not tools.pr.cancelled:
                tools.next_layer()
                obs = tools.observe()
                if obs.label == "ok":
                    report = tools.function_report(current=obs.measured)
                    if report["continue_ok"]:
                        continue
                    obs.label = "function_risk"                  # looks close to plan, but breaks a feature
                    tools.believe_measured(obs)
                else:
                    report = tools.function_report()
                if acknowledged and report["continue_ok"] and obs.deviation_mm <= acknowledged + regrow_mm:
                    continue                                     # same, already-accepted deviation
                paused = gate.pause_for_alarm()
                options = [dict(id="resume" if paused else "continue",
                                effect="keep printing the original plan"),
                           dict(id="stop", effect="stop the print; it is scrapped and must be restarted")]
                situation = dict(observation=dict(layer=obs.layer, z=obs.z, label=obs.label,
                                                  deviation_mm=round(obs.deviation_mm, 2), paused=paused),
                                 function_if_continue=report, cost=tools.cost(), status=tools.status())
                rec = decider.decide(situation, options)
                action = gate.decide_and_act(rec, options, _summary(situation, rec))
                cv2.imwrite(str(run / "frames" / f"{obs.layer:04d}.png"), obs.image)
                event = dict(t=time.time(), situation=situation, recommendation=rec, action=action,
                             frame=f"frames/{obs.layer:04d}.png")
                f.write(json.dumps(event, default=str) + "\n")
                f.flush()
                events.append(event)
                log(f"layer {obs.layer}: {obs.label} {situation['observation']['deviation_mm']:.2f} mm -> {action} ({rec['source']})")
                if action == "stop":
                    outcome = "stopped"
                    break
                acknowledged = max(acknowledged or 0.0, obs.deviation_mm)
        except PrintEnded as e:
            outcome = f"ended on the printer: {e}"
        except KeyboardInterrupt:
            outcome = "agent interrupted (the print itself is not affected)"
    transcript = getattr(decider, "transcript", None)
    if transcript:
        (run / "llm_transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    summary = dict(outcome=outcome, layers=tools.pr.n_done, alarms=len(events), decider=decider.name)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"{outcome} after {tools.pr.n_done} layers, {len(events)} decision(s); see {run}")
    return dict(summary, run=str(run), events=events)
