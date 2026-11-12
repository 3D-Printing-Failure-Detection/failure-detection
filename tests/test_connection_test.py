"""The laptop-to-printer connection checklist, against a fake printer and a real address with
nothing listening."""
import json

from recovery.connection_test import MockPrinter, RealPrinter, run


def _quiet(*a):
    pass


def test_healthy_mock_passes_every_step(tmp_path):
    res = run(MockPrinter(), str(tmp_path), commands=True, ask=lambda p: "yes", log=_quiet)
    assert res["ok"] and set(res["steps"]) == {"network", "status", "camera", "files", "commands"}
    assert (tmp_path / res["dir"].split("/")[-1] / "camera.jpg").exists()
    assert json.loads(open(res["dir"] + "/report.json").read())["ok"] is True


def test_a_failed_step_is_reported_with_a_hint(tmp_path):
    lines = []
    res = run(MockPrinter(["camera"]), str(tmp_path), log=lines.append)
    assert not res["ok"] and not res["steps"]["camera"]["ok"] and res["steps"]["files"]["ok"]
    assert any("hint: Close Bambu Studio" in l for l in lines)


def test_commands_need_a_running_print_and_a_yes(tmp_path):
    idle = MockPrinter()
    idle.state = "IDLE"
    res = run(idle, str(tmp_path), commands=True, ask=lambda p: "yes", log=_quiet)
    assert not res["steps"]["commands"]["ok"] and "start a print first" in res["steps"]["commands"]["detail"]
    res = run(MockPrinter(), str(tmp_path), commands=True, ask=lambda p: "no", log=_quiet)
    assert "commands" not in res["steps"]                      # declined: nothing sent


def test_unreachable_printer_stops_after_the_network_step(tmp_path):
    res = run(RealPrinter("127.0.0.1", "SERIAL", "12345678"), str(tmp_path), log=_quiet)
    assert not res["ok"] and list(res["steps"]) == ["network"]
