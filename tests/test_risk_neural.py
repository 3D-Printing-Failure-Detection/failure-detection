from types import SimpleNamespace

import pytest
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.ops import unary_union

from recovery.geometry.adapt import Plan
from recovery.geometry import features as F
from recovery.geometry.spec import Verdict
from recovery.loop.decide import DecisionContext
from recovery.loop.risk_neural import RiskAwareNeuralPolicy, PreventiveFitMixin
from recovery.loop import risk_neural
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


@pytest.mark.parametrize('alternative_passes', [False, True])
def test_joint_forecasts_require_a_passing_alternative_before_excluding_a_plan(monkeypatch, alternative_passes):
    layer = Layer(.25, Point(0, 0).buffer(15).difference(Point(0, 0).buffer(5)))
    policy = object.__new__(RiskAwareNeuralPolicy)
    policy.joint_forecast = True
    policy.analytical_candidate_checks = 0
    policy.spec = {'holes': [dict(center=[0, 0], diameter=[8.5, 11.5], tol_center=1.)]}
    policy.allow_adapt, policy.use_fea, policy.reserve_fit = True, False, True
    policy.h, policy.max_correction_per_layer_mm = .25, .25
    policy.nominal = [layer] * 4
    history = [layer]
    policy.state = SimpleNamespace(model=object(), observations=history, residuals=[], uncertainty_mm=.2)

    def candidate(plan, ok=True, score=.8):
        return dict(name=plan.name, plan=plan, executable=True,
                    verdict=Verdict(ok, [] if ok else ['forecast fails'], {}),
                    margin=score, robust_margin=score, fit_clearance_mm=score,
                    fit_robust_clearance_mm=score)

    primary = [candidate(Plan('continue', [layer]*3, 1.), False),
               candidate(Plan('learned inverse', [layer]*3, 1.))]
    monkeypatch.setattr(PreventiveFitMixin, 'candidates', lambda *args, **kwargs: primary)
    monkeypatch.setattr(policy, '_targets', lambda known: [(Plan('target', [layer]*3, 1.), {})])
    monkeypatch.setattr(policy, '_evaluate', lambda known, plan, *args: candidate(plan))
    analytical_state = SimpleNamespace(compensate=lambda layers: layers)
    analytical = SimpleNamespace(state=analytical_state,
        _evaluate=lambda known, plan, *args: candidate(plan,
            alternative_passes and plan.name.startswith('analytical inverse:'), score=.3))
    monkeypatch.setattr(risk_neural, 'RiskAwareContourPolicy', lambda *args, **kwargs: analytical)
    before = layer.region.wkb
    result = policy.candidates(history, policy.nominal, DecisionContext(3))
    assert policy.analytical_candidate_checks == 2
    assert history == [layer] and layer.region.wkb == before
    assert primary[1]['verdict'].ok  # independent assessment never mutates the original
    if alternative_passes:
        assert not result[1]['verdict'].ok
        assert result[2]['verdict'].ok and result[2]['both_forecasts_pass']
        assert result[2]['fit_robust_clearance_mm'] == .3
    else:
        assert result is primary


@pytest.mark.parametrize('analytical_ok,clearance,recovery_ok,recovery_clearance,expected', [
    (False, -.1, True, .4, 'recover'),  # intervene while the prefix is still repairable
    (True, .4, True, .6, 'continue'),  # both forecasts already cover observed uncertainty
    (False, -.1, False, -.2, 'continue'),  # no jointly passing alternative: preserve neural choice
    (True, .1, True, .2, 'continue'),  # improvement is below half a camera pixel
    (True, .1, True, .4, 'recover'),  # low analytical clearance has a measurable remedy
])
def test_joint_forecast_checks_a_confident_neural_continuation(monkeypatch,
        analytical_ok, clearance, recovery_ok, recovery_clearance, expected):
    layer = Layer(.25, Point(0, 0).buffer(15).difference(Point(0, 0).buffer(5)))
    policy = object.__new__(RiskAwareNeuralPolicy)
    policy.joint_forecast, policy.analytical_candidate_checks = True, 0
    policy.spec = {'holes': [dict(center=[0, 0], diameter=[8.5, 11.5], tol_center=1.)]}
    policy.allow_adapt, policy.use_fea, policy.reserve_fit = True, False, True
    policy.h, policy.max_correction_per_layer_mm = .25, .25
    policy.nominal = [layer] * 4
    history = [layer]
    policy.state = SimpleNamespace(model=object(), observations=history, residuals=[],
        uncertainty_mm=.3, resolution_mm=.25, compensate=lambda layers: layers)

    def candidate(plan, ok=True, slack=.8):
        return dict(name=plan.name, plan=plan, predicted=plan.layers, executable=True,
            verdict=Verdict(ok, [] if ok else ['fit threatened'], {}),
            margin=slack, robust_margin=slack, fit_clearance_mm=slack,
            fit_robust_clearance_mm=slack, command_change_fraction=.01)

    primary = [candidate(Plan('continue', [layer]*3, 1.))]
    monkeypatch.setattr(PreventiveFitMixin, 'candidates', lambda *args, **kwargs: primary)
    monkeypatch.setattr(policy, '_targets', lambda known: [(Plan('target', [layer]*3, 1.), {})])
    monkeypatch.setattr(policy, '_evaluate', lambda known, plan, *args: candidate(plan))
    analytical = SimpleNamespace(state=SimpleNamespace(compensate=lambda layers: layers),
        _evaluate=lambda known, plan, *args: candidate(plan,
            analytical_ok if plan.name == 'continue' else recovery_ok,
            clearance if plan.name == 'continue' else recovery_clearance))
    monkeypatch.setattr(risk_neural, 'RiskAwareContourPolicy', lambda *args, **kwargs: analytical)
    before = layer.region.wkb
    result = policy.candidates(history, policy.nominal, DecisionContext(3))
    assert policy.analytical_candidate_checks >= 1
    assert primary[0]['verdict'].ok and 'fit_at_risk' not in primary[0]
    assert layer.region.wkb == before and policy.state.observations is history
    if expected == 'continue':
        assert result is primary
    else:
        assert result[0]['fit_at_risk']
        assert any(c['name'] != 'continue' and c['both_forecasts_pass'] for c in result)
        assert policy.pick_candidates(result, DecisionContext(3)).action == 'adapt'
