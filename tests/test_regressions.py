from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, box

from recovery import camera
from recovery.gcode import gcode_to_layers
from recovery.geometry.adapt import Plan
from recovery.geometry.spec import Verdict
from recovery.loop.decide import RulePolicy
from recovery.parts import Part
from recovery.types import Layer


def test_nested_island_is_not_erased_by_surrounding_ring():
    ring = Point(0, 0).buffer(15).difference(Point(0, 0).buffer(10))
    island = Point(0, 0).buffer(8)
    image = camera.rasterise(island.union(ring))
    center = camera.CAM.size // 2
    assert image[center, center] == 255
    assert image[center, center + 9 * 4] == 0


def test_csg_nested_and_outer_height_ranges_agree(tmp_path):
    texts = ["  - cut: {circle: {r: 4}, z: [1, 4]}",
             "  - cut: {circle: {r: 4}}\n    z: [1, 4]"]
    stacks = []
    for i, text in enumerate(texts):
        path = tmp_path / f"part{i}.yaml"
        path.write_text("height: 4\nlayer_h: 1\ngeometry:\n  - add: {circle: {r: 6}}\n" + text)
        stacks.append(Part.from_yaml(path).layers())
    assert all(a.region.equals(b.region) for a, b in zip(*stacks))
    assert not stacks[0][0].region.interiors
    assert stacks[0][1].region.interiors


def test_gcode_width_and_height_are_modal_per_extrusion():
    text = "M83\n;LAYER_CHANGE\nG1 Z0.2\n;WIDTH:0.4\nG1 X10 E1\n;LAYER_CHANGE\nG1 Z0.4\n;WIDTH:1.2\nG1 Y10 E1\n"
    layers = gcode_to_layers(text, origin=(0, 0))
    assert [l.z for l in layers] == [.2, .4]
    assert layers[0].region.bounds[3] - layers[0].region.bounds[1] == pytest.approx(.4)
    assert layers[1].region.bounds[2] - layers[1].region.bounds[0] == pytest.approx(1.2)


def test_gcode_relative_xy_and_fractional_coordinates():
    text = "M83\n;LAYER_CHANGE\n;Z:0.2\nG1 X10 Y10\nG91\nG1 X.5 Y+1.5 E1\n"
    lay = gcode_to_layers(text, origin=(0, 0))[0]
    assert lay.region.centroid.x == pytest.approx(10.25)
    assert lay.region.centroid.y == pytest.approx(10.75)


def test_fea_filter_never_advertises_untested_candidates(monkeypatch):
    import recovery.fea
    calls = []
    def stress(layers, *args):
        calls.append(layers)
        ok = len(calls) == 1
        return SimpleNamespace(ok=ok, violations=[] if ok else ["unsafe"])
    monkeypatch.setattr(recovery.fea, "stress_test", stress)
    layer = Layer(.25, box(0, 0, 5, 5))
    policy = RulePolicy({"loads": [{}]}, [layer], .25, use_fea=True)
    candidates = [dict(plan=Plan(str(i), [layer]), verdict=Verdict(True)) for i in range(3)]
    policy._stress_filter([], candidates)
    assert len(calls) == 3
    assert [c["verdict"].ok for c in candidates] == [True, False, False]


def test_live_idle_timeout_does_not_invent_a_printed_layer():
    from recovery.printer.bambu import BambuPrinter
    printer = BambuPrinter.__new__(BambuPrinter)
    printer.timeout = 0
    printer._layer = 0
    printer.printed = []
    with pytest.raises(TimeoutError):
        printer.print_layer()
    assert printer._layer == 0 and printer.printed == []


def test_live_pause_requires_acknowledgement():
    from recovery.printer.bambu import BambuPrinter
    printer = BambuPrinter.__new__(BambuPrinter)
    printer.link = SimpleNamespace(command=lambda *a: None, wait_state=lambda *a, **k: False)
    with pytest.raises(TimeoutError):
        printer.pause()


def test_fe_axial_bar_matches_analytic_displacement():
    from recovery.fea.material import Material
    from recovery.fea.model import Model
    from recovery.fea.voxel import VoxelGrid
    # Zero Poisson ratio makes uniform axial strain an exact patch test.
    material = Material("test", 1000, 1000, 0, 0, 50, 50, 50, 0)
    model = Model(VoxelGrid(np.ones((2, 2, 4), bool), 1, 1, 0, 0), material)
    bottom = np.isclose(model.xyz[:, 2], 0)
    top = np.isclose(model.xyz[:, 2], 4)
    fixed = np.repeat(bottom, 3)
    force = np.zeros(model.ndof)
    # Consistent traction loads: sum unit face forces at each top element node.
    for elem in np.nonzero(model.ei[:, 2] == 3)[0]:
        force[model.enode[elem, 4:] * 3 + 2] += 1.
    u, converged, _ = model.solve(force, fixed)
    expected = force.sum() * 4 / (4 * 1000)  # F L / A E
    assert converged
    assert np.allclose(u.reshape(-1, 3)[top, 2], expected, atol=1e-7)
