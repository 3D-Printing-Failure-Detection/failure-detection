"""Agentic pipeline: printer tools, deciders (rules and the LLM slot), the approval gate and the
loop, rehearsed on the simulated printer with the real test parts and hand dents."""
import json

import pytest

from recovery.agent import ActionGate, AutoApprover, LLMDecider, PrinterTools, RuleDecider, TerminalApprover, run_agent
import numpy as np

from recovery.agent.rehearse import PlannedPauseSim
from recovery.geometry.spec import check
from recovery.parts import get_part


class ScriptedWatcher:
    """Stands in for the camera's failure detector (models/p1s_layer.LayerWatcher): what it reports for
    each finished layer index is scripted, so the agent's decisions can be checked exactly."""

    def __init__(self, script=None):
        self.script = script or {}

    def check(self, prev, cur, plan, part_xy, yaw_deg=0.0):
        kind, px = self.script.get(len(plan) - 1, ("ok", 0))
        return dict(kind=kind, flagged_px=px, type_prob={})


class FastSim(PlannedPauseSim):
    """The simulated printer without rendering (the scripted watcher does not look at frames)."""

    def snapshot(self):
        return np.zeros((720, 1280, 3), np.uint8)


def _run(part, script, tmp_path, decider=None, approver=None, seed=1, failure_fn=None):
    p = get_part(f"parts/print/{part}/{part}.yaml")
    nom = p.layers()
    pr = FastSim(nom, failure_fn(nom) if failure_fn else None, seed=seed)
    tools = PrinterTools(pr, p, ScriptedWatcher(script))
    gate = ActionGate(tools, approver or AutoApprover(), log=lambda *a: None)
    res = run_agent(tools, decider or RuleDecider(), gate, out_dir=str(tmp_path), log=lambda *a: None)
    return p, nom, pr, res


SAG = {k: ("collapse", 120) for k in (29, 30, 31)}             # a wall sags from layer 29: alarm at 31


def test_one_or_two_odd_layers_do_not_raise_an_alarm(tmp_path):
    script = {12: ("warp", 200), 30: ("collapse", 150), 31: ("collapse", 150)}
    _, _, pr, res = _run("two-hole-block", script, tmp_path)
    assert res["events"] == [] and res["outcome"] == "finished"


def test_what_the_camera_always_flags_on_this_print_is_not_an_alarm(tmp_path):
    script = {k: ("warp", 80) for k in range(60)}                    # e.g. a calibration that is a bit off
    script.update({k: ("warp", 400) for k in range(40, 60)})        # until a real problem appears
    _, _, pr, res = _run("two-hole-block", script, tmp_path)
    assert [e["situation"]["observation"]["layer"] for e in res["events"]] == [42]


def test_clean_print_raises_no_alarm(tmp_path):
    _, _, pr, res = _run("two-hole-block", None, tmp_path)
    assert res["outcome"] == "finished" and res["events"] == []


def test_a_starting_failure_is_paused_once_then_resumed(tmp_path):
    script = {k: ("collapse", 120 + k - 29) for k in range(29, 60)}   # stays about the same size
    _, _, pr, res = _run("two-hole-block", script, tmp_path)
    assert [e["action"] for e in res["events"]] == ["resume"] and res["outcome"] == "finished"
    assert res["events"][0]["situation"]["observation"]["flagged_px"] == 122      # third layer in a row


def test_a_failure_that_keeps_growing_is_stopped(tmp_path):
    script = {k: ("collapse", 120 + 40 * (k - 29)) for k in range(29, 60)}
    _, _, pr, res = _run("two-hole-block", script, tmp_path)
    assert [e["action"] for e in res["events"]] == ["resume", "stop"] and pr.cancelled
    assert res["events"][1]["situation"]["observation"]["grown"]


@pytest.mark.parametrize("kind", ["detach", "shift"])
def test_detach_and_shift_are_stopped_at_once(tmp_path, kind):
    _, _, pr, res = _run("water-tray", {k: (kind, 900) for k in range(20, 60)}, tmp_path)
    assert [e["action"] for e in res["events"]] == ["stop"] and pr.cancelled and pr.n_done == 23


def test_a_camera_problem_is_reported_and_the_print_goes_on(tmp_path):
    class Flaky(ScriptedWatcher):
        def check(self, prev, cur, plan, part_xy, yaw_deg=0.0):
            if len(plan) in (5, 6):
                raise TimeoutError("no camera frame")
            return super().check(prev, cur, plan, part_xy, yaw_deg)
    p = get_part("parts/print/water-tray/water-tray.yaml")
    pr = FastSim(p.layers(), None, seed=1)
    tools = PrinterTools(pr, p, Flaky())
    logs = []
    res = run_agent(tools, RuleDecider(), ActionGate(tools, AutoApprover(), log=lambda *a: None),
                    out_dir=str(tmp_path), log=logs.append)
    assert res["outcome"] == "finished" and res["layers_unchecked"] == 2 and res["events"] == []
    assert any("could not check" in l for l in logs)


def test_event_log_is_written(tmp_path):
    _run("two-hole-block", SAG, tmp_path)
    lines = (tmp_path.glob("*/events.jsonl").__next__()).read_text().splitlines()
    ev = json.loads(lines[0])
    assert set(ev) >= {"situation", "recommendation", "action", "frames"}


def test_llm_choice_is_used_when_valid(tmp_path):
    calls = []

    def complete(system, user):
        calls.append(json.loads(user))
        return '{"action": "stop", "reason": "be conservative"}'
    dec = LLMDecider(complete)
    _, _, pr, res = _run("two-hole-block", SAG, tmp_path, decider=dec)
    assert pr.cancelled and res["events"][0]["recommendation"]["source"] == "llm"
    assert {o["id"] for o in calls[0]["allowed_actions"]} == {"resume", "stop"}
    assert "function_if_continue" in calls[0]["situation"]


@pytest.mark.parametrize("reply", ['{"action": "replan everything"}', "not json", None])
def test_bad_llm_reply_falls_back_to_rules_visibly(tmp_path, reply):
    def complete(system, user):
        if reply is None:
            raise ConnectionError("no server")
        return reply
    _, _, pr, res = _run("two-hole-block", SAG, tmp_path, decider=LLMDecider(complete))
    rec = res["events"][0]["recommendation"]
    assert rec["source"] == "llm->rules" and rec["action"] == "resume" and not pr.cancelled


def test_person_can_override_the_recommendation(tmp_path):
    answers = iter(["9", "2"])                                 # an invalid key, then [2] stop
    approver = TerminalApprover(ask=lambda prompt: next(answers), show=lambda *a: None)
    _, _, pr, res = _run("two-hole-block", SAG, tmp_path, approver=approver)
    assert res["events"][0]["recommendation"]["action"] == "resume" and res["events"][0]["action"] == "stop"
    assert pr.cancelled


def test_real_run_refuses_auto_approve():
    from recovery.cli import main
    with pytest.raises(SystemExit, match="sim only"):
        main(["agent", "--part", "parts/print/two-hole-block/two-hole-block.yaml", "--gcode", "x.gcode.3mf",
              "--ip", "1", "--serial", "s", "--code", "c", "--auto-approve"])


def test_reply_parser_ignores_a_thinking_block():
    from recovery.agent.deciders import parse_reply
    text = '<think>maybe {"action": "resume"} ... no</think>\n{"action": "stop", "reason": "hole too small"}'
    assert parse_reply(text) == {"action": "stop", "reason": "hole too small"}


def test_auto_chain_without_key_or_server_falls_back_to_rules(monkeypatch):
    from recovery.agent.cli import OPTIONS, SAMPLE, make_decider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dec = make_decider("auto", endpoint="http://127.0.0.1:9")      # nothing listens there
    out = dec.decide(SAMPLE, OPTIONS)
    assert out["action"] == "stop" and out["source"] == "local:qwen3:1.7b->rules"
    assert "error" in dec.full_transcript[0]


def test_claude_failure_falls_back_to_local_then_rules(monkeypatch):
    from recovery.agent.deciders import LLMDecider

    def broken(system, user):
        raise RuntimeError("no credits")
    local = LLMDecider(lambda s, u: '{"action": "resume", "reason": "cosmetic"}', name="local")
    dec = LLMDecider(broken, name="claude", fallback=local)
    from recovery.agent.cli import OPTIONS, SAMPLE_OK
    out = dec.decide(SAMPLE_OK, OPTIONS)
    assert out["action"] == "resume" and out["source"] == "claude->local"
    assert [r["decider"] for r in dec.full_transcript] == ["claude", "local"]



# ---------------------------------------------------------------- container dent test at a planned pause
def _dent_test(change, tmp_path, decider=None, approver=None):
    from recovery.agent.loop import DentTest
    from recovery.agent.measure import ScriptedMeasure
    from recovery.agent.rehearse import PlannedPauseSim
    p = get_part("parts/print/water-tray/water-tray.yaml")
    nom = p.layers()
    pr = PlannedPauseSim(nom, None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    test = DentTest(person=lambda: pr.user_change(change), measure=ScriptedMeasure(change))
    gate = ActionGate(tools, approver or AutoApprover(), log=lambda *a: None)
    res = run_agent(tools, decider or RuleDecider(), gate, out_dir=str(tmp_path), dent_test=test, log=lambda *a: None)
    return p, nom, pr, res


@pytest.mark.parametrize("change, action, works", [
    ("none", "resume", True),
    ("dent 0 -10 1", "resume", True),          # cosmetic: continue as planned
    ("dent 0 -10 3", "correct", True),         # next layer would hang in air: new layers fade back
    ("dent 15 0 4", "correct", True),          # same on the right wall
    ("poke 5 -10", "stop", None),              # wall open: already-printed layers cannot be fixed
])
def test_tray_dent_test_decides_once_and_the_part_ends_as_predicted(tmp_path, change, action, works):
    p, nom, pr, res = _dent_test(change, tmp_path)
    assert [e["action"] for e in res["events"]] == [action]            # the agent is asked exactly once
    if works is None:
        assert pr.cancelled and pr.n_done == 30
    else:
        truth = check(pr.printed, nom, p.spec, p.layer_h)               # the shape actually printed
        assert res["outcome"] == "finished" and truth.ok, truth.violations


def test_dent_test_saves_before_after_and_change_frames(tmp_path):
    _, _, _, res = _dent_test("dent 0 -10 3", tmp_path)
    ev = res["events"][0]
    assert set(ev["frames"]) == {"before", "after", "change"}
    assert ev["situation"]["observation"]["camera_change"]["changed_fraction"] > 0
    assert ev["situation"]["correction"]["fade_layers"] and ev["options"] == ["resume", "correct", "stop"]


def test_printed_layers_are_not_judged_for_hanging_over_air(tmp_path):
    # a 3 mm bend of the printed top layers is not itself a failure; only new layers printed over it are
    from recovery.agent.rehearse import PlannedPauseSim
    p = get_part("parts/print/water-tray/water-tray.yaml")
    pr = PlannedPauseSim(p.layers(), None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    while tools.next_layer() != "paused":
        pass
    tools.believe_top(pr.user_change("dent 0 -10 3"))
    report = tools.function_report(future=tools.correction()["layers"])
    assert report["continue_ok"]


@pytest.mark.parametrize("change, leaks", [("dent 0 -10 1", False), ("dent 0 -10 3", True), ("dent 15 0 4", True)])
def test_resuming_over_a_deep_dent_leaks_between_layers(change, leaks):
    # each layer is a closed ring, but past the wall thickness the next layer's cavity sits over open air
    p = get_part("parts/print/water-tray/water-tray.yaml")
    pr = PlannedPauseSim(p.layers(), None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    while tools.next_layer() != "paused":
        pass
    tools.believe_top(pr.user_change(change))
    def container(future):
        return next(f for f in tools.function_report(future=future)["features"] if f["feature"].startswith("holds"))
    resume = container(None)
    assert resume["ok"] != leaks and ("open air" in resume["detail"]) == leaks
    assert container(tools.correction()["layers"])["ok"]                # the correction stays over the dented wall


def test_person_can_type_the_dent(tmp_path):
    from recovery.agent.measure import ManualMeasure
    answers = iter(["dent 0 -10", "dent 0 -10 3"])               # first answer is incomplete
    shown = []
    m = ManualMeasure(ask=lambda prompt: next(answers), show=shown.append)
    from recovery.agent.rehearse import PlannedPauseSim
    p = get_part("parts/print/water-tray/water-tray.yaml")
    pr = PlannedPauseSim(p.layers(), None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    while tools.next_layer() != "paused":
        pass
    new, described = m(tools)
    assert described == "dent 0 -10 3" and len(new) == 5 and shown


def test_bambu_reports_a_planned_pause_once():
    import threading
    from recovery.printer.bambu import BambuPrinter

    class Link:
        def __init__(self):
            self.state, self.sent, self.lock = dict(gcode_state="PAUSE", layer_num=31), [], threading.Lock()

        def get(self, k, d=None):
            return self.state.get(k, d)

        def command(self, *a):
            self.sent.append(a)

    pr = BambuPrinter.__new__(BambuPrinter)
    pr.plan, pr.printed, pr.link, pr.notify_pause = [object()] * 60, [], Link(), True
    pr._layer, pr._offset, pr.timeout, pr.settle, pr.cancelled = 29, 0, 1.0, 0.0, False
    assert pr.print_layer() == "paused" and pr.n_done == 30            # layer 31 is about to print
    pr.resume()
    assert pr.link.sent == [("resume",)]
    with pytest.raises(TimeoutError):                                   # still PAUSE in the report: not again
        pr.print_layer()


# ---------------------------------------------------------------- correction G-code for the real printer
def test_gap_fill_closes_the_middle_of_the_trays_thin_wall():
    import shapely
    from recovery.gcode import gap_fill_paths, layer_paths
    wall = get_part("parts/print/water-tray/water-tray.yaml").layers()[40]
    paths = layer_paths(wall.region, 40, w=0.42, loops=2)
    lines = [shapely.LineString(q).buffer(0.21) for q in paths]
    before = wall.region.difference(shapely.union_all(lines).buffer(0.01)).area / wall.region.area
    gaps = gap_fill_paths(wall.region, paths, 0.42)
    lines += [shapely.LineString(q).buffer(g / 2) for q, g in gaps]
    after = wall.region.difference(shapely.union_all(lines).buffer(0.01)).area / wall.region.area
    extruded = (sum(shapely.LineString(q).length * 0.42 for q in paths)
                + sum(shapely.LineString(q).length * g for q, g in gaps)) / wall.region.area
    assert before > 0.2 and after < 0.01 and 0.9 < extruded < 1.1     # filled, not over-extruded


def test_correction_gcode_retracts_and_drops_the_bed_for_every_travel():
    from recovery.printer.bambu import FOOTER, resume_gcode
    nom = get_part("parts/print/water-tray/water-tray.yaml").layers()
    gc = resume_gcode(nom[30:32], 0.2, 30, (128.0, 128.0))
    assert gc.endswith(FOOTER)
    lines = gc[:-len(FOOTER)].splitlines()                             # the layers, without the end/park code
    travels = [i for i, l in enumerate(lines) if l.startswith("G0 ")]
    assert travels
    for i in travels:
        assert lines[i - 1].startswith("G1 Z") and lines[i + 1].startswith("G1 Z")      # up, move, down
        assert float(lines[i - 1].split("Z")[1].split()[0]) > float(lines[i + 1].split("Z")[1].split()[0])
    assert sum(" E-" in l for l in lines) == len(travels) - 1                          # all but the first
    assert any(l == ";TYPE:Gap infill" for l in lines)


def test_simulator_gcode_is_unchanged_by_default():
    from recovery.gcode import write_layers
    nom = get_part("parts/print/water-tray/water-tray.yaml").layers()
    text, _ = write_layers(nom[30:31], 0.2)
    assert " E-" not in text and "Gap infill" not in text and text.count("G1 Z") == 1


def test_bed_temperature_follows_the_plate_in_use():
    from recovery.gcode import print_settings
    cfg = ("; nozzle_temperature = 215,215\n; hot_plate_temp = 55,55\n; textured_plate_temp = 65,65\n"
           "; curr_bed_type = Textured PEI Plate\n")
    assert print_settings(cfg) == dict(nozzle=215, bed=65)
    assert print_settings(cfg.replace("Textured PEI Plate", "High Temp Plate"))["bed"] == 55


def test_correction_gcode_and_preview_are_saved_before_approval(tmp_path):
    import os
    from recovery.gcode import gcode_to_layers
    from recovery.geometry.ops import iou
    seen = []

    class Inspecting(AutoApprover):                    # what a person could open at the prompt
        def choose(self, recommendation, options, summary):
            paths = [w.rstrip(",") for w in summary.split() if w.endswith((".gcode,", ".png"))]
            seen.extend(os.path.exists(x) for x in paths)
            return super().choose(recommendation, options, summary)
    p, nom, pr, res = _dent_test("dent 0 -10 3", tmp_path, approver=Inspecting())
    assert seen == [True] * 5                          # before, after, change frames + G-code and preview
    fix = res["events"][0]["situation"]["correction"]
    parsed = gcode_to_layers(open(fix["gcode"]).read())
    assert len(parsed) == 30 and iou(parsed[0].region, pr.plan[30].region) > 0.95


def test_resume_test_sends_the_original_remaining_layers(tmp_path):
    from recovery.agent.rehearse import PlannedPauseSim
    p = get_part("parts/print/water-tray/water-tray.yaml")
    nom = p.layers()
    pr = PlannedPauseSim(nom, None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    gate = ActionGate(tools, AutoApprover(), log=lambda *a: None)
    res = run_agent(tools, RuleDecider(), gate, out_dir=str(tmp_path), monitor=False, resume_test=True,
                    log=lambda *a: None)
    assert [e["action"] for e in res["events"]] == ["restart"] and res["outcome"] == "finished"
    assert all(a is b for a, b in zip(pr.plan, nom))                      # the plan itself is unchanged
    assert check(pr.printed, nom, p.spec, p.layer_h).ok


def test_if_someone_resumes_on_the_printer_the_agent_sends_nothing(tmp_path):
    from recovery.agent.loop import DentTest
    from recovery.agent.measure import ScriptedMeasure
    from recovery.agent.rehearse import PlannedPauseSim
    p = get_part("parts/print/water-tray/water-tray.yaml")
    pr = PlannedPauseSim(p.layers(), None, seed=1, pause_at=30)
    tools = PrinterTools(pr, p)
    sent = []
    tools.resume = lambda: sent.append("resume")

    def dent_then_resume_on_screen():
        pr.user_change("dent 0 -10 1")
        pr.paused = False                                   # someone pressed resume on the printer
    test = DentTest(person=dent_then_resume_on_screen, measure=ScriptedMeasure("dent 0 -10 1"))
    gate = ActionGate(tools, AutoApprover(), log=lambda *a: None)
    res = run_agent(tools, RuleDecider(), gate, out_dir=str(tmp_path), monitor=False, dent_test=test,
                    log=lambda *a: None)
    ev = res["events"][0]
    assert ev["options"][0] == "continue" and ev["action"] == "continue" and sent == []
    assert res["outcome"] == "finished"


def test_camera_checks_are_opt_in_on_the_real_printer(tmp_path):
    """Real printer: typed measurement and no every-layer watching unless asked; --sim uses both."""
    import argparse
    from recovery.agent.cli import add_arguments, resolve_modes
    ckpt = tmp_path / "layer"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"")
    p = argparse.ArgumentParser()
    add_arguments(p)
    base = ["--part", "parts/mug.yaml", "--layer-model", str(ckpt)]
    a = p.parse_args(base)
    assert resolve_modes(a) is False and a.measure == "manual"
    a = p.parse_args(base + ["--sim"])
    assert resolve_modes(a) is True and a.measure == "sideview"
    a = p.parse_args(base + ["--monitor", "on", "--measure", "sideview"])
    assert resolve_modes(a) is True and a.measure == "sideview"
    a = p.parse_args(["--part", "parts/mug.yaml", "--sim", "--layer-model", str(tmp_path / "missing")])
    assert resolve_modes(a) is False


def test_dent_depth_is_typed_and_evidence_is_shown():
    from recovery.agent.loop import _summary
    from recovery.agent.measure import ask_depth, describe_evidence
    answers, shown = iter(["", "deep", "-1", "2.5"]), []
    assert ask_depth("front", 0, -10, ask=lambda _: next(answers), show=shown.append) == 2.5 and len(shown) == 3
    missed = describe_evidence(dict(kind="none", changed_px=3, min_px=5, p_dent=0.5, p_poke=0.5, wall=None, p_wall=None))
    assert "check the images" in missed
    sure = describe_evidence(dict(kind="poke", changed_px=400, min_px=5, p_dent=0.1, p_poke=0.9, wall="front", p_wall=0.97))
    assert "poke 90%" in sure and "check the images" not in sure
    situation = dict(observation=dict(label="dent test", layer=29, z=6.0, described="camera (side view): none",
                                      evidence=dict(frames=dict(before="/r/b.png", after="/r/a.png", change="/r/c.png"),
                                                    detector=missed)),
                     cost=dict(filament_used_mm=1, filament_total_mm=2, fraction_done=0.5),
                     function_if_continue=dict(features=[]))
    text = _summary(situation, dict(source="rules", action="resume", reason="x"))
    assert "/r/b.png" in text and "/r/c.png" in text and "check the images" in text
