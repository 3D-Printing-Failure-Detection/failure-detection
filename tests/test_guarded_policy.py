import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import box

from recovery.geometry import features as F
from recovery.geometry.adapt import Plan
from recovery.geometry.spec import Verdict
from recovery.loop.controller import Controller
from recovery.loop.decide import DecisionContext
from recovery.loop.guarded import GuardedPolicy
from recovery.loop.recoverability import correction_budget, intervention_tradeoff
from recovery.models.classical import Detection
from recovery.sim import SimPrinter
from recovery.types import Layer


def test_notebook_reachability_examples_and_exact_boundary():
    result = correction_budget(2, .25, 15, .2)
    assert result["required_layers"] == 8
    assert result["transition_height_mm"] == pytest.approx(1.6)
    assert result["spare_layers"] == 7 and result["nominal_reachable"]
    assert not correction_budget(2, .125, 15, .2)["nominal_reachable"]
    assert correction_budget(0, .25, 0, .2)["required_layers"] == 0
    assert correction_budget(1.1, .1, 11, .2)["required_layers"] == 11
    assert correction_budget(1.10000001, .1, 11, .2)["required_layers"] == 12


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True])
def test_reachability_rejects_invalid_inputs(value):
    with pytest.raises(ValueError):
        correction_budget(value, .25, 15, .2)


def test_notebook_economics_extends_to_continue_and_preserves_unknown_probability():
    # Notebook: restart costs 5 h, recovery 2 h + .25 h intervention, p=.8.
    result = intervention_tradeoff(0, 2, .25, 5, p_continue=0, p_recovery=.8)
    assert result["required_success_gain"] == pytest.approx(.45)
    assert result["expected_savings"] == pytest.approx(1.75)
    # Continuing is already likely to work; recovery's same .8 is not a gain.
    result = intervention_tradeoff(2, 2, .25, 5, p_continue=.8, p_recovery=.8)
    assert result["expected_savings"] == pytest.approx(-.25)
    assert not result["recovery_economically_preferred"]
    unknown = intervention_tradeoff(2, 2, .25, 5)
    assert unknown["required_success_gain"] == pytest.approx(.05)
    assert unknown["expected_savings"] is None and unknown["recovery_economically_preferred"] is None
    with pytest.raises(ValueError):
        intervention_tradeoff(2, 2, .25, 5, p_continue=.8)


def test_bounded_correction_reaches_target_at_assumed_rate():
    layers = [Layer(.25 * (i + 1), box(-5, -5, 5, 5)) for i in range(10)]
    known = [Layer(.25, translate(layers[0].region, 2, 0))]
    policy = GuardedPolicy({}, layers, .25)
    targets = list(policy._targets(known))
    plan, budget = next((p, b) for p, b in targets if p.name == "bounded return")
    assert budget["required_layers"] == 8
    offsets = [2.] + [p.region.centroid.x for p in plan.layers]
    assert np.max(np.abs(np.diff(offsets))) <= .25 + 1e-8
    assert offsets[8] == pytest.approx(0., abs=1e-8)
    # No full return fits two remaining layers; partial/held shapes still exist.
    short = GuardedPolicy({}, layers[:3], .25)
    assert any(p.name == "bounded partial correction" and not b["nominal_reachable"]
               for p, b in short._targets(known))


def test_exhausted_budget_continues_a_passing_print_without_replanning(monkeypatch):
    layers = [Layer(.25 * (i + 1), box(-5, -5, 5, 5)) for i in range(4)]
    policy = GuardedPolicy({}, layers, .25)
    monkeypatch.setattr(policy, "_targets", lambda _: pytest.fail("must not build an unnecessary intervention"))
    result = policy.decide(layers[:1], layers, Detection("functional_risk", .4, {}),
                           context=DecisionContext(adapts_remaining=0))
    assert result.action == "continue"


def test_exhausted_budget_does_not_continue_an_unrepairable_part():
    layers = [Layer(.25 * (i + 1), box(-5, -5, 5, 5)) for i in range(4)]
    spec = {"container": {"base_h": .5, "capacity_mm3": [0, 1000]}}
    damaged = Layer(.25, layers[0].region.difference(box(-1, -1, 1, 1)))
    policy = GuardedPolicy(spec, layers, .25)
    result = policy.decide([damaged], layers, Detection("functional_risk", .4, {}),
                           context=DecisionContext(adapts_remaining=0))
    assert result.action == "scrap" and "continuation fails" in result.reason


def test_rejects_out_of_bounds_candidate_and_selects_feasible_alternative(monkeypatch):
    layers = [Layer(.25, box(0, 0, 10, 10)), Layer(.5, box(0, 0, 12, 10))]
    policy = GuardedPolicy({"material_volume_mm3": [35, 50]}, layers, .25)
    too_wide = Plan("out of bounds", [Layer(.5, box(-100, 0, 100, 1))])
    feasible = Plan("fits bed", [Layer(.5, box(0, 0, 7, 10))])
    monkeypatch.setattr(policy, "_targets", lambda _: [(too_wide, None), (feasible, None)])
    monkeypatch.setattr(policy.state, "compensate", lambda layers: layers)

    def validate(plan):
        if any(l.region.bounds[0] < -20 or l.region.bounds[2] > 20 for l in plan):
            raise ValueError("outside bed")

    result = policy.decide(layers[:1], layers, Detection("warp", 2, {}),
                           context=DecisionContext(1, validate))
    assert result.action == "adapt" and result.plan.name == "compensate:fits bed"
    rejected = next(c for c in result.considered if c["name"] == "compensate:out of bounds")
    assert not rejected["executable"] and not rejected["ok"]
    assert "outside bed" in rejected["violations"]


def test_controller_keeps_measuring_repeated_warnings_after_budget_exhaustion():
    layers = [Layer(.25 * (i + 1), box(-5, -5, 5, 5)) for i in range(7)]
    printer = SimPrinter(layers)
    policy = GuardedPolicy({}, layers, .25)

    class Segmenter:
        def measure(self, *_):
            return printer.expected_layer().region

    class Detector:
        def detect(self, *_):
            return Detection("functional_risk", .4, {})

    result = Controller(printer, Segmenter(), Detector(), policy, {}, layers, .25, max_adapts=0).run()
    assert result.functional and len(result.events) == 7
    assert all(e.action == "continue" for e in result.events)
    assert len(policy.state.observations) == 7


def test_overhang_cache_respects_nominal_reference_without_retaining_previous_layer():
    layer = Layer(.5, box(0, 0, 10, 10))
    previous = Layer(.25, box(0, 0, 5, 10))
    # Same actual pair: allowed by a nominal overhang, unsupported by a flat design.
    assert F.excess_overhang(layer, previous, layer, previous) == 0
    nominal = Layer(.5, box(0, 0, 10, 10))
    nominal_previous = Layer(.25, box(0, 0, 5, 10))
    assert F.excess_overhang(layer, previous, nominal, nominal_previous) == 0
    supported_nominal = Layer(.25, box(0, 0, 10, 10))
    assert F.excess_overhang(layer, previous, nominal, supported_nominal) > 6
    weak = weakref.ref(previous)
    token = previous.cache_token
    del previous
    assert weak() is None
    assert token is not Layer(.25, box(0, 0, 5, 10)).cache_token


def test_forecast_evaluation_precedes_current_measurement_and_does_not_update_model(monkeypatch):
    from recovery.eval.mujoco_eval import ObservedSegmenter
    from recovery.models.classical import ClassicalSegmenter
    layer = Layer(.25, box(-5, -5, 5, 5))
    policy = GuardedPolicy({}, [layer] * 5, .25)
    previous = Layer(.25, translate(layer.region, 1, 0))
    for _ in range(4):
        policy.state.observe(layer, previous)
    actual = Layer(.25, translate(layer.region, 4, 0))
    printer = SimpleNamespace(printed=[actual], expected_layer=lambda: layer,
                              failure=SimpleNamespace(k0=4), n_done=5)
    segmenter = ObservedSegmenter(printer)
    segmenter.policy = policy
    monkeypatch.setattr(ClassicalSegmenter, "measure", lambda *_: actual.region)
    segmenter.measure(np.zeros((8, 8, 3), dtype=np.uint8))
    assert segmenter.forecast_scores[0]["forecast_mm"] > 2.
    assert segmenter.forecast_scores[0]["command_mm"] > 3.
    assert len(policy.state.observations) == 4
