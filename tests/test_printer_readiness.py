"""What has to work before the first runs on the real P1S: parsing Bambu Studio output, and a
watch-only printer backend that can never command the printer."""
import threading
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


def test_watch_runs_end_to_end_without_commanding(tmp_path, monkeypatch):
    import json
    from recovery import camera, live

    class FakeBambu:
        def __init__(self, ip, serial, code, planned, layer_h, bed_camera, observe_only=False, **kw):
            assert observe_only, "watch must build an observe-only printer"
            self.plan, self.printed, self.skipped_layers, self.last_frame = list(planned), [], 0, None
            self.link = FakeLink(gcode_state="RUNNING", total_layer_num=len(planned), layer_num=1)
            self.rng = np.random.default_rng(0)

        n_done = property(lambda self: len(self.printed))
        n_total = property(lambda self: len(self.plan))

        def print_layer(self):
            self.printed.append(self.plan[len(self.printed)])

        def camera_image(self):
            self.last_frame = camera.render(self.printed[-3:], self.rng)
            return self.last_frame

        def expected_layer(self):
            return self.plan[len(self.printed) - 1]

        def close(self):
            pass

    monkeypatch.setattr(live, "BambuPrinter", FakeBambu)
    monkeypatch.setattr(live.BedCamera, "load", classmethod(lambda cls, p: None))
    f = tmp_path / "p.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/plate_1.gcode", _bambu_file(3))
    run = live.run_watch(None, str(f), "ip", "serial", "code", "calib.npz", out=str(tmp_path / "w"),
                         log=lambda *a: None)
    rows = [json.loads(l) for l in (run / "layers.jsonl").read_text().splitlines()]
    assert [r["layer"] for r in rows] == [0, 1, 2] and all("label" in r for r in rows)
    assert len(list((run / "frames").glob("*_raw.jpg"))) == 3
    meta = json.loads((run / "meta.json").read_text())
    assert meta["layers_observed"] == 3 and meta["printer_total_layer_num"] == 3
