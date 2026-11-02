from types import SimpleNamespace

import pytest
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.ops import unary_union

from recovery.geometry.adapt import Plan
from recovery.geometry import features as F
from recovery.geometry.spec import Verdict
from recovery.loop.decide import DecisionContext
from recovery.loop.risk_neural import RiskAwareNeuralPolicy
from recovery.models.classical import Detection
from recovery.types import Layer


@pytest.mark.parametrize('continued_diameter,recovered_diameter,budget,expected', [
    (8.7, 9.2, 3, 'adapt'),   # significant new clearance before the fit is lost
    (8.7, 8.8, 3, 'continue'),  # improvement is smaller than camera resolution
    (9.5, 10., 3, 'continue'),  # continuation already covers the uncertainty
    (8.7, 9.2, 0, 'continue'),  # no adaptation budget remains
])
def test_preventive_fit_recovery_respects_evidence_and_budget(monkeypatch, continued_diameter, recovered_diameter, budget, expected):
    # Exercise decision semantics independently of a particular trained set of
    # weights; the printing experiments separately verify real model inference.
    policy = object.__new__(RiskAwareNeuralPolicy)
    policy.spec = dict(holes=[dict(center=[0, 0], diameter=[8.5, 11.5], tol_center=1.)])
    policy.allow_adapt = True
    layer = Layer(.25, Point(0, 0).buffer(15).difference(Point(0, 0).buffer(5)))
    policy.nominal = [layer] * 4
    policy.state = SimpleNamespace(uncertainty_mm=.3, resolution_mm=.25, compensate=lambda layers: layers)
    monkeypatch.setattr(policy, '_targets', lambda known: [(Plan('test recovery', [layer]*3, 1.), {})])

    def evaluate(known, plan, context, current_plan, budget=None):
        diameter = continued_diameter if plan.name == 'continue' else recovered_diameter
        return dict(name=plan.name, plan=plan, predicted=plan.layers, executable=True,
                    verdict=Verdict(True, [], {'hole_[0, 0]_d': diameter, 'hole_[0, 0]_offset_mm': .2}),
                    margin=.5, robust_margin=-1., command_change_fraction=.01)

    monkeypatch.setattr(policy, '_evaluate', evaluate)
    decision = policy.decide([layer], policy.nominal, Detection('functional_risk', .9, {}),
                             context=DecisionContext(budget))
    assert decision.action == expected
    assert len(decision.plan.layers) == 3


def test_clearance_reserve_is_bounded_and_never_changes_the_prefix():
    policy = object.__new__(RiskAwareNeuralPolicy)
    policy.spec = dict(holes=[dict(center=[0, 0], diameter=[8.5, 11.5], tol_center=1.)])
    policy.reserve_fit = True
    policy.h = .25
    policy.max_correction_per_layer_mm = .25
    region = Point(0, 0).buffer(15).difference(Point(0, 0).buffer(5))
    policy.nominal = [Layer(.25*(i+1), region) for i in range(8)]
    observed = Layer(.25, translate(region, .9, 0))
    before = observed.region.wkb
    plans = list(policy._targets([observed]))
    hold, reserve = plans[0][0], plans[-1][0]
    assert reserve.name == 'bounded hole clearance'
    assert observed.region.wkb == before
    assert [l.z for l in reserve.layers] == [l.z for l in policy.nominal[1:]]
    for i, (original, expanded) in enumerate(zip(hold.layers, reserve.layers), 1):
        old_void = unary_union(F.hole_near(original, [0, 0], 11.5))
        new_void = unary_union(F.hole_near(expanded, [0, 0], 11.5))
        assert old_void.difference(new_void).area < 1e-6
        assert new_void.difference(old_void.buffer(i*.25+1e-5, quad_segs=8)).area < 1e-5
