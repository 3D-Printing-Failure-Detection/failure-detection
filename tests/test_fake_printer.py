"""The simulated P1S (recovery/printer/fake_p1s.py) talks to the real printer code (recovery/printer/bambu.py)
over the same connections a real P1S uses, and `watch` runs against it end to end."""
from recovery import p1s_view  # noqa: F401  (first: selects the EGL backend before MuJoCo loads)

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("paho.mqtt.client")

from recovery.alarm import AlarmRule, first_alarm  # noqa: E402

TOWER = "tests/fixtures/unseen_parts/thin-tower.yaml"      # 150 layers; tests start near the top


def test_alarm_rule_learns_a_baseline_then_needs_layers_in_a_row():
    r = AlarmRule(warmup=2, confirm=2)
    assert [r.update(k, px) for k, px in [("warp", 40), ("ok", 0)]] == [False, False]   # baseline: median(40, 0)
    assert r.baseline == 20
    assert r.update("warp", 15) is False           # not more than the baseline
    assert r.update("warp", 30) is False           # 1 in a row
    assert r.update("ok", 0) is False              # broken
    assert [r.update("warp", 30), r.update("collapse", 60), r.update("warp", 90)] == [False, True, True]


def test_first_alarm_on_a_logged_print():
    layers = [dict(layer=i, kind="ok" if i < 6 else "warp", flagged_px=0 if i < 6 else 50) for i in range(10)]
    assert first_alarm(layers, warmup=3, confirm=3)["layer"] == 8
    assert first_alarm(layers[:7], warmup=3, confirm=3) is None


@pytest.fixture
def sim():
    from recovery.parts import get_part
    from recovery.printer.fake_p1s import FakeP1S
    p = FakeP1S(get_part(TOWER), mqtt_port=0, camera_port=0, layer_s=0.4, start_layer=140, fps=4.0, log=lambda *a: None)
    yield p
    p.close()


def _printer(sim, observe_only=True):
    from recovery.parts import get_part
    from recovery.printer.bambu import BambuPrinter
    part = get_part(TOWER)
    return BambuPrinter("127.0.0.1", sim.serial, sim.code, part.layers(), part.layer_h, origin=part.origin,
                        observe_only=observe_only, settle_s=0.1, timeout_s=60, ports=sim.ports)


def _wait(cond, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_the_printer_code_reads_status_and_frames_from_the_simulated_printer(sim):
    pr = _printer(sim)
    try:
        assert _wait(lambda: pr.link.get("total_layer_num") == sim.n)
        sim.heartbeat().start()
        pr.print_layer()                                   # catches up to where the print is
        assert pr.n_done >= 140 and pr.skipped_layers >= 139
        frame = pr.snapshot()
        assert frame.shape == (720, 1280, 3) and frame.std() > 3
        with pytest.raises(RuntimeError, match="observe_only"):
            pr.pause()                                     # watch never sends a command
    finally:
        pr.close()


def test_pause_resume_and_stop_reach_the_simulated_printer(sim):
    from recovery.printer.bambu import PrintEnded
    sim.layer_s = 1.0
    pr = _printer(sim, observe_only=False)
    try:
        assert _wait(lambda: pr.link.get("gcode_state") == "IDLE")
        sim.heartbeat().start()
        assert _wait(lambda: pr.link.get("gcode_state") == "RUNNING")
        pr.pause()
        n = sim.layer_num
        time.sleep(1.5)
        assert sim.state == "PAUSE" and sim.layer_num == n  # nothing prints while paused
        pr.resume()
        assert _wait(lambda: sim.layer_num > n, 120), f"render took {getattr(sim, 'render_s', None)} s"
        pr.cancel()
        assert _wait(lambda: sim.state == "IDLE")
        assert _wait(lambda: pr.link.get("gcode_state") == "IDLE")
        pr._seen_running = True
        with pytest.raises(PrintEnded):
            pr.print_layer()
    finally:
        pr.close()


def test_a_wrong_access_code_gets_no_camera_frames(sim):
    from recovery.printer.bambu import BambuCamera
    cam = BambuCamera("127.0.0.1", "wrong", sim.ports["camera"])
    try:
        with pytest.raises(TimeoutError):
            cam.grab(timeout=3)
    finally:
        cam.close()


def test_watch_end_to_end_against_the_simulated_printer(sim, tmp_path):
    from recovery import live
    model = "runs/p1s_layer" if Path("runs/p1s_layer/model.pt").exists() else None
    sim.layer_s = 2.0
    sim.heartbeat().start(wait_for_watcher=True)
    run = live.run_watch(TOWER, None, "127.0.0.1", sim.serial, sim.code, None, model, out=str(tmp_path),
                         ports=sim.ports, view_port=0, log=lambda *a: None)
    rows = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines()]
    meta = json.loads((run / "meta.json").read_text())
    # a slow checker skips ahead rather than falling behind, so not every layer is guaranteed
    assert meta["layers_observed"] == len(rows) >= 5 and rows[-1]["layer"] == sim.n - 1
    assert len(list((run / "marked").glob("*.jpg"))) == len(rows)
    if model:
        assert all("label" in r for r in rows[1:])
        assert all(r["label"] != "unchecked" for r in rows[1:]), "the plan outline should be in view"
