"""Agentic pipeline: printer tools, deciders (rules and the LLM slot), the approval gate and the
loop, rehearsed on the simulated printer with the real test parts and hand dents."""
import json

import pytest

from recovery.agent import ActionGate, AutoApprover, LLMDecider, PrinterTools, RuleDecider, TerminalApprover, run_agent
from recovery.agent.rehearse import hand_dent, hand_poke
from recovery.geometry.spec import check
from recovery.models.classical import ClassicalDetector, ClassicalSegmenter
from recovery.parts import get_part
from recovery.sim import SimPrinter


def _run(part, failure_fn, tmp_path, decider=None, approver=None, seed=1):
    p = get_part(f"parts/print/{part}/{part}.yaml")
    nom = p.layers()
    pr = SimPrinter(nom, failure_fn(nom) if failure_fn else None, seed=seed)
    tools = PrinterTools(pr, p, ClassicalSegmenter(), ClassicalDetector())
    gate = ActionGate(tools, approver or AutoApprover(), log=lambda *a: None)
    res = run_agent(tools, decider or RuleDecider(), gate, out_dir=str(tmp_path), log=lambda *a: None)
    return p, nom, pr, res


def test_clean_print_raises_no_alarm(tmp_path):
    _, _, pr, res = _run("two-hole-block", None, tmp_path)
    assert res["outcome"] == "finished" and res["events"] == []


def test_cosmetic_dent_is_paused_then_resumed_and_part_works(tmp_path):
    p, nom, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path)
    assert [e["action"] for e in res["events"]] == ["resume"]
    assert res["outcome"] == "finished" and check(pr.printed, nom, p.spec, p.layer_h).ok


def test_small_dent_that_narrows_a_hole_is_caught_by_the_function_check(tmp_path):
    # the outline moves < 1 mm (below the detector's alarm), but the hole no longer takes the bolt
    _, _, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 15, -5, 1.0, 29), tmp_path)
    ev = res["events"][0]
    assert ev["situation"]["observation"]["label"] == "function_risk" and ev["action"] == "stop"
    assert pr.cancelled and pr.n_done == 30                    # stopped at the dent layer itself
    bad = [f["feature"] for f in ev["situation"]["function_if_continue"]["features"] if not f["ok"]]
    assert bad == ["hole at [15, 0] (dia 5.1-6.0 mm)"]


def test_tray_poke_through_is_stopped(tmp_path):
    _, _, pr, res = _run("water-tray", lambda n: hand_poke(n, 5, -10, 29), tmp_path)
    assert pr.cancelled and res["events"][-1]["action"] == "stop"


def test_event_log_is_written(tmp_path):
    _, _, _, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path)
    lines = (tmp_path.glob("*/events.jsonl").__next__()).read_text().splitlines()
    ev = json.loads(lines[0])
    assert set(ev) >= {"situation", "recommendation", "action", "frames"}


def test_llm_choice_is_used_when_valid(tmp_path):
    calls = []

    def complete(system, user):
        calls.append(json.loads(user))
        return '{"action": "stop", "reason": "be conservative"}'
    dec = LLMDecider(complete)
    _, _, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path, decider=dec)
    assert pr.cancelled and res["events"][0]["recommendation"]["source"] == "llm"
    assert {o["id"] for o in calls[0]["allowed_actions"]} == {"resume", "stop"}
    assert "function_if_continue" in calls[0]["situation"]


@pytest.mark.parametrize("reply", ['{"action": "replan everything"}', "not json", None])
def test_bad_llm_reply_falls_back_to_rules_visibly(tmp_path, reply):
    def complete(system, user):
        if reply is None:
            raise ConnectionError("no server")
        return reply
    _, _, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path, decider=LLMDecider(complete))
    rec = res["events"][0]["recommendation"]
    assert rec["source"] == "llm->rules" and rec["action"] == "resume" and not pr.cancelled


def test_person_can_override_the_recommendation(tmp_path):
    answers = iter(["9", "2"])                                 # an invalid key, then [2] stop
    approver = TerminalApprover(ask=lambda prompt: next(answers), show=lambda *a: None)
    _, _, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path, approver=approver)
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
    tools = PrinterTools(pr, p, ClassicalSegmenter(), ClassicalDetector())
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
