"""What has to work before the first runs on the real P1S: parsing Bambu Studio output, and a
watch-only printer backend that can never command the printer."""
import threading
import time
import zipfile

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from recovery.gcode import gcode_to_layers, inspect_gcode, layer_paths, splice_gcode
from recovery.geometry.ops import iou
from recovery.parts import Part
from recovery.printer.bambu import BambuCamera, BambuPrinter, PrintEnded

PART = box(-20, -20, 20, 20).difference(Point(0, 0).buffer(4, quad_segs=32))


def _num(v):
    s = f"{v:.4f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)


def _bambu_layer(z, w=0.42, cx=128.0, cy=128.0):
    """One Bambu-Studio-style layer: 40 mm square with an 8 mm hole, walls, sparse infill every
    4 mm, a skirt, and numbers written without a leading zero ("E.0123")."""
    def moves(path):
        out = [f"G1 X{_num(path[0, 0] + cx)} Y{_num(path[0, 1] + cy)} F12000"]
        for a, b in zip(path[:-1], path[1:]):
            out.append(f"G1 X{_num(b[0] + cx)} Y{_num(b[1] + cy)} E{_num(np.hypot(*(b - a)) * 0.033)}")
        return out

    g = ["; CHANGE_LAYER", f"; Z_HEIGHT: {z}", "; LAYER_HEIGHT: 0.2", f"; LINE_WIDTH: {w}",
         f"G1 Z{_num(z)}", "; FEATURE: Skirt"]
    g += moves(np.array(Point(0, 0).buffer(35, quad_segs=16).exterior.coords))
    for feat, path in layer_paths(PART, 0, w, loops=2, tagged=True):
        if feat == "Solid infill":
            continue
        g.append("; FEATURE: " + feat)
        g += moves(path)
        g.append("G1 E-.8 F1800")                                   # retract (negative E)
    g.append("; FEATURE: Sparse infill")
    for s in np.arange(-26, 26, 4.0):
        seg = PART.buffer(-0.8).intersection(LineString([(s - 30, -30), (s + 30, 30)]))
        for piece in getattr(seg, "geoms", [seg]):
            if not piece.is_empty and piece.length > 0.5:
                g += moves(np.array(piece.coords))
    return "\n".join(g) + "\n"


def _bambu_file(n=3):
    return "; total layer number: %d\nM83\n" % n + "".join(_bambu_layer(round(0.2 * (i + 1), 2)) for i in range(n))


def test_bambu_markers_parse_to_solid_layers_with_only_the_real_hole():
    L = gcode_to_layers(_bambu_file(2))
    assert [round(l.z, 2) for l in L] == [0.2, 0.4]
    r = L[0].region
    assert iou(r, PART) > 0.97                       # solid between infill lines, skirt ignored
    assert r.geom_type == "Polygon" and len(r.interiors) == 1


def test_inspect_reports_markers_and_outline_rebuild():
    rep = inspect_gcode(_bambu_file(3))
    assert rep["markers"]["CHANGE_LAYER"] == 3 and rep["markers"]["Z_HEIGHT:"] == 3
    assert rep["layers_with_material"] == 3 and rep["slicer_total_layers"] == 3
    assert rep["rebuild"] == {"from_walls": 3} and rep["empty_layers"] == []
    assert "Skirt" in rep["features"]


def test_splice_finds_bambu_layer_markers():
    out, _ = splice_gcode(_bambu_file(2), 1, [], 0.2)
    assert out.count("; CHANGE_LAYER") == 1


def test_part_from_bambu_3mf(tmp_path):
    f = tmp_path / "plate.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/plate_1.gcode", _bambu_file(4))
    part = Part.from_gcode(f)
    assert len(part.layers()) == 4 and part.layer_h == pytest.approx(0.2) and part.dropped_layers == 0
    assert part.origin == pytest.approx((128.0, 128.0), abs=0.05)


def test_unparseable_gcode_gives_a_clear_error(tmp_path):
    f = tmp_path / "x.gcode"
    f.write_text("G28\nG1 X10 Y10\n")
    with pytest.raises(ValueError, match="check-gcode"):
        Part.from_gcode(f)


def test_cli_check_gcode(tmp_path, capsys):
    from recovery.cli import main
    f = tmp_path / "p.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/plate_1.gcode", _bambu_file(3))
    main(["check-gcode", str(f), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "looks usable" in out and (tmp_path / "p_layers.png").exists()


# ---------------------------------------------------------------- watch-only printer backend
class FakeLink:
    def __init__(self, **state):
        self.state, self.sent, self.lock = dict(state), [], threading.Lock()

    def get(self, k, default=None):
        return self.state.get(k, default)

    def command(self, *a):
        self.sent.append(a)


def _printer(observe_only, n=10, **state):
    pr = BambuPrinter.__new__(BambuPrinter)          # no network: fill in what print_layer uses
    pr.plan, pr.printed, pr.link = [object()] * n, [], FakeLink(**state)
    pr.observe_only, pr.skipped_layers, pr._seen_running = observe_only, 0, False
    pr._layer, pr._offset, pr.timeout, pr.settle, pr._layer_done_t = 0, 0, 2.0, 0.0, 0.0
    pr.cancelled = False
    return pr


def test_observe_only_never_commands_the_printer():
    pr = _printer(True)
    for call in (pr.pause, pr.resume, pr.cancel, lambda: pr.replace_remaining([])):
        with pytest.raises(RuntimeError, match="observe_only"):
            call()
    assert pr.link.sent == []


def test_observe_only_skips_ahead_when_layers_finished_while_busy():
    pr = _printer(True, gcode_state="RUNNING", layer_num=6)
    pr.print_layer()
    assert pr.n_done == 5 and pr.skipped_layers == 4


def test_acting_mode_still_refuses_missed_layers():
    pr = _printer(False, gcode_state="RUNNING", layer_num=6)
    with pytest.raises(RuntimeError, match="missed"):
        pr.print_layer()


def test_stopping_the_print_on_the_screen_ends_the_wait():
    pr = _printer(False, gcode_state="RUNNING", layer_num=1)
    pr._seen_running = True
    pr.link.state["gcode_state"] = "IDLE"
    with pytest.raises(PrintEnded):
        pr.print_layer()


def test_idle_before_the_print_starts_is_not_an_end():
    pr = _printer(True, gcode_state="IDLE", layer_num=0)
    pr.timeout = 0.6
    with pytest.raises(TimeoutError):                 # keeps waiting for the job to start
        pr.print_layer()


def test_camera_waits_for_a_frame_newer_than_the_layer():
    cam = BambuCamera.__new__(BambuCamera)
    cam.frame, cam.frame_t = np.zeros((2, 2, 3), np.uint8), 100.0
    with pytest.raises(TimeoutError):
        cam.grab(timeout=0.3, newer_than=200.0)
    cam.frame_t = 201.0
    assert cam.grab(timeout=0.3, newer_than=200.0).shape == (2, 2, 3)


class _FakeWatchBambu:
    """Observe-only printer: three layers, a blank camera frame after each."""

    def __init__(self, ip, serial, code, planned, layer_h, observe_only=False, **kw):
        assert observe_only, "watch must build an observe-only printer"
        self.plan, self.printed, self.skipped_layers = list(planned), [], 0
        self.link = FakeLink(gcode_state="RUNNING", total_layer_num=len(planned), layer_num=1)

    n_done = property(lambda self: len(self.printed))
    n_total = property(lambda self: len(self.plan))

    def print_layer(self):
        self.printed.append(self.plan[len(self.printed)])

    def snapshot(self):
        return np.zeros((72, 128, 3), np.uint8)

    def close(self):
        pass


def _watch_file(tmp_path, n):
    f = tmp_path / "p.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/plate_1.gcode", _bambu_file(n))
    return f


def test_watch_checks_every_layer_with_the_camera_model_and_never_commands(tmp_path, monkeypatch):
    import json
    from recovery import live
    from recovery.models import p1s_layer

    seen = []

    class FakeWatcher:
        def __init__(self, ckpt, cam=None):
            pass

        def check(self, prev, cur, plan, part_xy, yaw_deg=0.0):
            seen.append(len(plan))
            return dict(kind="warp" if len(plan) >= 4 else "ok", flagged_px=50 * (len(plan) >= 4), type_prob={})

    monkeypatch.setattr(live, "BambuPrinter", _FakeWatchBambu)
    monkeypatch.setattr(p1s_layer, "LayerWatcher", FakeWatcher)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.pt").write_bytes(b"")
    run = live.run_watch(None, str(_watch_file(tmp_path, 6)), "ip", "serial", "code", None, str(model),
                         out=str(tmp_path / "w"), warmup=1, confirm=2, log=lambda *a: None)
    rows = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines()]
    # the first layer has nothing to compare with; then the agent's rule: 1 layer of baseline, 2 in a row
    assert "label" not in rows[0] and seen == [2, 3, 4, 5, 6]
    assert [r["label"] for r in rows[1:]] == ["ok", "ok", "warp", "warp", "warp"]
    assert [r["alarm"] for r in rows[1:]] == [False, False, False, True, True]
    assert len(list((run / "frames").glob("*.jpg"))) == 6 and len(list((run / "marked").glob("*.jpg"))) == 6
    meta = json.loads((run / "meta.json").read_text())
    assert meta["layers_observed"] == 6 and meta["first_alarm"]["layer"] == 4 and meta["printer_total_layer_num"] == 6


def test_watch_without_the_model_saves_frames_only(tmp_path, monkeypatch):
    import json
    from recovery import live
    monkeypatch.setattr(live, "BambuPrinter", _FakeWatchBambu)
    run = live.run_watch(None, str(_watch_file(tmp_path, 2)), "ip", "serial", "code", None,
                         str(tmp_path / "missing"), out=str(tmp_path / "w"), log=lambda *a: None)
    rows = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines()]
    assert [r["layer"] for r in rows] == [0, 1] and not any("label" in r for r in rows)
    assert len(list((run / "frames").glob("*.jpg"))) == 2
    assert json.loads((run / "meta.json").read_text())["layer_model"] is None


def test_dent_test_started_mid_print_jumps_to_the_current_layer():
    pr = _printer(False, gcode_state="RUNNING", layer_num=15, total_layer_num=10)
    pr.plan = [object()] * 60
    pr.notify_pause, pr.skip_ahead, pr.log = True, True, lambda *a: None
    pr.print_layer()
    assert pr.n_done == 14


def test_dent_test_waits_through_the_last_prints_end_state():
    shown = []
    pr = _printer(False, gcode_state="FINISH", layer_num=0)
    pr.notify_pause, pr.timeout, pr.log = True, 0.6, shown.append
    with pytest.raises(TimeoutError):                 # waits for the print to start instead of failing
        pr.print_layer()
    assert "Start the print" in shown[0]


def test_upload_does_not_wait_for_a_tls_shutdown_the_printer_never_sends():
    import io
    from recovery.printer.bambu import _BambuFTPS

    class Conn:
        sent = b""
        def sendall(self, b): Conn.sent += b
        def unwrap(self): raise TimeoutError("The read operation timed out")
        def __enter__(self): return self
        def __exit__(self, *a): return False

    ftp = _BambuFTPS.__new__(_BambuFTPS)
    ftp.voidcmd = lambda cmd: "200"
    ftp.transfercmd = lambda cmd, rest=None: Conn()
    ftp.voidresp = lambda: "226 Transfer complete"
    assert ftp.storbinary("STOR x.gcode", io.BytesIO(b"G1 X1\n" * 5000)) == "226 Transfer complete"
    assert len(Conn.sent) == 6 * 5000


def test_replacement_job_asks_the_person_to_start_it_if_the_printer_does_not(monkeypatch):
    import threading
    import recovery.printer.bambu as B
    uploads, shown = [], []
    monkeypatch.setattr(B, "upload_ftps", lambda ip, code, name, data: uploads.append(name))
    pr = _printer(False, gcode_state="PAUSE", layer_num=30)
    pr.plan, pr._layer, pr.ip, pr.code, pr.log = [object()] * 60, 29, "ip", "code", shown.append
    pr.link.lock, pr.link.wait_state = threading.Lock(), None

    def wait_state(states, timeout=60, poll=0.05):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if pr.link.state.get("gcode_state") in states:
                return True
            time.sleep(poll)
        return False

    def command(cmd, param=""):
        pr.link.sent.append((cmd, param))
        if cmd == "stop":
            pr.link.state["gcode_state"] = "IDLE"

    def person_starts_it():                       # the gcode_file command did nothing; the person starts it
        while not any("Start it from the printer screen" in m for m in shown):
            time.sleep(0.02)
        pr.link.state.update(gcode_state="RUNNING", layer_num=30)

    pr.link.wait_state, pr.link.command = wait_state, command
    threading.Thread(target=person_starts_it, daemon=True).start()
    pr._restart("G1\n", 29, [object()] * 31, start_wait_s=0.3, person_wait_s=5)
    assert len(uploads) == 1 and ("gcode_file", f"/sdcard/{uploads[0]}") in pr.link.sent
    assert pr._offset == 0 and len(pr.plan) == 60


def test_replacement_job_parks_the_head_at_the_end():
    from recovery.printer.bambu import FOOTER
    lines = FOOTER.split()
    assert "G28" not in FOOTER and lines.index("Z10") < lines.index("X65") < lines.index("M104")


def test_dent_test_finishes_even_if_the_layer_count_drifted_after_a_restart():
    pr = _printer(False, gcode_state="FINISH", layer_num=31)
    pr.plan, pr._layer, pr._seen_running = [object()] * 60, 40, True
    pr.notify_pause, pr.skip_ahead, pr.log = True, True, lambda *a: None
    pr.print_layer()
    assert pr.n_done == 60
