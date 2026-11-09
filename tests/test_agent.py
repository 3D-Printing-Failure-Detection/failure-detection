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
    assert set(ev) >= {"situation", "recommendation", "action", "frame"}


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
    answers = iter(["s"])                                      # recommendation is resume; person stops
    approver = TerminalApprover(ask=lambda prompt: next(answers), show=lambda *a: None)
    _, _, pr, res = _run("two-hole-block", lambda n: hand_dent(n, 0, -5, 2.0, 29), tmp_path, approver=approver)
    assert res["events"][0]["recommendation"]["action"] == "resume" and res["events"][0]["action"] == "stop"
    assert pr.cancelled


def test_real_run_refuses_auto_approve():
    from recovery.cli import main
    with pytest.raises(SystemExit, match="sim only"):
        main(["agent", "--part", "parts/print/two-hole-block/two-hole-block.yaml", "--gcode", "x.gcode.3mf",
              "--ip", "1", "--serial", "s", "--code", "c", "--auto-approve"])
