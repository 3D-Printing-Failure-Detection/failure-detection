import numpy as np
import pytest

from recovery import camera
from recovery.deform import random_failure
from recovery.gcode import gcode_to_layers, write_gcode
from recovery.geometry.ops import iou
from recovery.geometry.spec import check
from recovery.models.classical import ClassicalDetector, ClassicalSegmenter
from recovery.parts import PART_NAMES, get_part
from recovery.run import run_scenario


@pytest.mark.parametrize("name", PART_NAMES)
def test_nominal_parts_meet_their_spec(name):
    p = get_part(name)
    L = p.layers()
    assert check(L, L, p.spec, p.layer_h).ok


def test_custom_part_from_yaml_runs_through_the_loop(tmp_path):
    f = tmp_path / "plate.yaml"
    f.write_text("""
name: plate
height: 6
layer_h: 0.25
geometry:
  - add: {rect: {size: [50, 24]}}
  - cut: {circle: {center: [0, 0], r: 3}}
min_wall_mm: 3.0
holes: [{center: [0, 0], diameter: [4.5, 7.0], tol_center: 1.0}]
""")
    p = get_part(str(f))
    L = p.layers()
    assert check(L, L, p.spec, p.layer_h).ok
    sc = run_scenario(str(f), "shift", seed=0)
    assert sc.result.outcome in ("finished", "scrapped")


def test_gcode_round_trip():
    p = get_part("spacer")
    L = p.layers()
    text, filament = write_gcode(L, p.layer_h)
    L2 = gcode_to_layers(text)
    assert len(L2) == len(L) and filament > 0
    assert min(iou(a.region, b.region) for a, b in zip(L, L2)) > 0.97


def test_segmenter_recovers_clean_layer():
    P = get_part("bracket").layers()
    rng = np.random.default_rng(0)
    k = 20
    img = camera.render([P[0]] + P[k - 3:k + 1], rng)
    m = ClassicalSegmenter().measure(img, camera.expected_mask(P[k]))
    assert ClassicalDetector().detect(img, P[k].region, m).label == "ok"


def test_detach_is_scrapped():
    sc = run_scenario("cup", "detach", seed=1)
    assert sc.result.outcome == "scrapped"


def test_collapse_is_recovered_in_place():
    sc = run_scenario("mug", "collapse", seed=3)       # sag that makes planned layers overhang
    assert sc.result.outcome == "finished" and sc.result.functional
    assert any(e.action == "adapt" for e in sc.result.events)


def test_failure_that_ignoring_survives_is_not_scrapped():
    sc = run_scenario("spacer", "warp", seed=2)
    assert sc.result.outcome == "finished"


def test_gcode_arc_parsing_and_origin():
    from recovery.gcode import gcode_to_layers
    # full-circle G2 of radius 10 around the bed centre, relative extrusion
    txt = ";LAYER_CHANGE\n;Z:0.2\n;WIDTH:0.42\nM83\nG1 X138 Y128\nG2 X138 Y128 I-10 J0 E5\n"
    L = gcode_to_layers(txt)
    assert len(L) == 1 and abs(L[0].region.centroid.x) < 0.5 and L[0].region.area > 0
    assert abs(L[0].region.bounds[2] - 10.2) < 0.4


def test_read_gcode_from_bambu_3mf(tmp_path):
    import zipfile
    from recovery.slicing import read_gcode_3mf
    f = tmp_path / "x.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/plate_1.gcode", "G1 X1\n")
    assert read_gcode_3mf(f) == "G1 X1\n"


def test_bed_camera_topdown_is_identity_for_a_straight_down_camera():
    import cv2
    from recovery.camera import BedCamera, CAM
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    # bed frame seen from straight above, 400 mm up, bed point (100,100) on the optical axis
    cam = BedCamera(K, np.zeros(5), np.array([np.pi, 0, 0]), np.array([-100.0, 100.0, 400.0]))
    H = cam.plane_to_image(0.0)
    p = H @ np.array([100.0, 100.0, 1.0])
    assert np.allclose(p[:2] / p[2], [320, 240], atol=1e-6)
    img = np.zeros((480, 640, 3), np.uint8)
    out = cam.top_down(img, 0.0, (100.0, 100.0))
    assert out.shape == (CAM.size, CAM.size, 3)


def test_heuristic_specgen_is_verified_on_the_nominal_part():
    from recovery.specgen import author_spec
    part = get_part(PART_NAMES[0])
    res = author_spec(part.layers(), part.layer_h, "", use_llm=False, h=2.0)
    assert res["source"] == "heuristic" and res["verified"], res["issues"]


def test_validate_spec_rejects_unknown_keys():
    from recovery.specgen import geometry_summary, validate_spec
    part = get_part(PART_NAMES[0])
    summ = geometry_summary(part.layers(), part.layer_h)
    assert validate_spec({"not_a_rule": 1}, summ)


def test_stress_test_reports_a_safety_factor():
    from recovery.fea import stress_test
    part = next((get_part(n) for n in PART_NAMES if get_part(n).spec.get("loads")), None)
    if part is None:
        pytest.skip("no part defines loads")
    rep = stress_test(part.layers(), part.spec, part.layer_h, 2.0)
    assert rep.cases and all(c.safety_factor > 0 for c in rep.cases)
