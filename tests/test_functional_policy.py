"""Behavioral checks for measured state, persistence and function-first selection."""
import numpy as np
from types import SimpleNamespace
from shapely.affinity import translate
from shapely.geometry import Point, box

from recovery.geometry.adapt import Plan
from recovery.geometry.ops import iou
from recovery.geometry.spec import check
from recovery.loop.controller import Controller
from recovery.loop.functional import FunctionalPolicy, functional_margin
from recovery.loop.observation import ObservationState
from recovery.models.classical import ClassicalDetector, Detection
from recovery.types import Layer


def test_controller_preserves_subthreshold_observations():
    nominal = [Layer(.25 * (i + 1), box(-5, -5, 5, 5)) for i in range(4)]
    measured = [Layer(l.z, translate(l.region, .3, 0)) for l in nominal]

    class Printer:
        def __init__(self):
            self.printed, self.plan, self.cancelled = [], nominal, False
            self.n_total = len(nominal)

        @property
        def n_done(self):
            return len(self.printed)

        def print_layer(self):
            self.printed.append(measured[self.n_done])

        def camera_image(self):
            return None

        def expected_layer(self):
            return nominal[self.n_done - 1]

    printer = Printer()

    class Segmenter:
        def measure(self, *_):
            return printer.printed[-1].region

    policy = FunctionalPolicy({}, nominal, .25)
    result = Controller(printer, Segmenter(), ClassicalDetector(), policy, {}, nominal, .25).run()
    assert result.functional and not result.events
    assert len(policy.state.observations) == 4
    assert all(abs(l.region.centroid.x - .3) < .01 for l in policy.state.observations)


def test_subthreshold_wall_loss_and_blocked_hole_raise_functional_alarm():
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    layer = Layer(.25, ring)
    policy = FunctionalPolicy({"min_wall_mm": 1.4}, [layer], .25)
    thin = Layer(.25, ring.buffer(-.45))
    det = ClassicalDetector().detect(None, ring, thin.region)
    assert det.label == "ok"
    _, det = policy.observe(layer, thin, det, 0)
    assert det.label == "functional_risk"

    plate = box(-20, -20, 20, 20).difference(Point(0, 0).buffer(.3))
    nominal = Layer(.25, plate)
    policy = FunctionalPolicy({"holes": [{"center": [0, 0], "diameter": [.4, 1.], "tol_center": .3}]}, [nominal], .25)
    filled = Layer(.25, box(-20, -20, 20, 20))
    _, det = policy.observe(nominal, filled, Detection("ok", .2, {}), 0)
    assert det.label == "functional_risk"
    assert "blocked" in det.info["reason"]


def test_prediction_persists_and_compensation_inverts_error():
    model = ObservationState()
    nominal = Layer(.25, Point(0, 0).buffer(10))
    for _ in range(4):
        model.observe(nominal, Layer(.25, translate(nominal.region, 1.5, -.5)))
    forecast = model.predict([nominal])[0]
    assert np.allclose(forecast.region.centroid.coords[0], [1.5, -.5], atol=.08)
    command = model.compensate([nominal])[0]
    assert np.allclose(command.region.centroid.coords[0], [-1.5, .5], atol=.08)
    assert iou(model.predict([command])[0].region, nominal.region) > .97
    # A second observation of a compensated command must still learn the plant
    # bias; comparing to the CAD here would incorrectly learn that it vanished.
    model.observe(command, Layer(.25, translate(command.region, 1.5, -.5)))
    assert np.allclose(model.displacement(np.array([[0., 0.]])), [[1.5, -.5]], atol=.1)


def test_margin_selection_prefers_function_over_fidelity_or_first_passing():
    nominal = [Layer(.25, box(0, 0, 10, 10))]
    marginal = [Layer(.25, box(0, 0, 8.01, 10))]
    spec = {"material_volume_mm3": [20., 30.]}
    policy = FunctionalPolicy(spec, nominal, .25)
    candidates = []
    for name, layers, fidelity in (("continue", marginal, 1.), ("safer", nominal, .8)):
        verdict = check(layers, nominal, spec, .25)
        margin = functional_margin(layers, nominal, spec, .25, verdict)
        candidates.append(dict(name=name, plan=Plan(name, layers, fidelity), verdict=verdict,
                               margin=margin, robust_margin=margin))
    decision = policy.pick(candidates)
    assert decision.action == "adapt" and decision.plan.name == "safer"


def test_predicted_future_does_not_erase_unrepairable_printed_prefix():
    ring = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(8))
    nominal = [Layer(.25 * (i + 1), ring) for i in range(3)]
    broken = Layer(.25, ring.difference(box(0, -1, 11, 1)))
    spec = {"container": {"base_h": 0, "capacity_mm3": [1, 1000]}}
    policy = FunctionalPolicy(spec, nominal, .25)
    policy.observe(nominal[0], broken, Detection("ok", .8, {}), 0)
    decision = policy.decide([broken], nominal, Detection("functional_risk", .8, {}))
    assert decision.action == "scrap"
    assert all(not c["ok"] for c in decision.considered)


def test_fea_checks_predicted_deposit_instead_of_compensating_commands(monkeypatch):
    import recovery.fea
    nominal = Layer(.25, box(0, 0, 10, 10))
    command = Layer(.25, translate(nominal.region, -2, 0))
    actual = Layer(.25, nominal.region.buffer(-1))
    seen = []

    def stress(layers, *_):
        seen.extend(layers)
        return SimpleNamespace(ok=False, violations=["predicted deposit too weak"])

    monkeypatch.setattr(recovery.fea, "stress_test", stress)
    policy = FunctionalPolicy({"loads": [{}]}, [nominal], .25, use_fea=True)
    candidate = dict(plan=Plan("compensate", [command]), predicted=[actual],
                     verdict=check([actual], [nominal], {}, .25))
    policy._stress_filter([], [candidate])
    assert seen == [actual]
    assert not candidate["verdict"].ok
