from copy import deepcopy
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from recovery.geometry.adapt import Plan
from recovery.geometry.spec import Verdict
from recovery.loop.camera_fallback import FallbackNeuralPolicy
from recovery.loop.decide import DecisionContext
from recovery.models.classical import Detection
from recovery.types import Layer
from tools.audit_neural_experiment import validate_damped_continuation


def observed_policy(enabled=True, bad_view=None):
    nominal = [Layer((n+1)*.25, box(-5, -5, 5, 5)) for n in range(10)]
    camera = SimpleNamespace()
    policy = FallbackNeuralPolicy({'present': [{'z': [0, 3], 'min_area_frac': .9}]},
        nominal, .25, checkpoint='unused-for-nonfits', forecast_scope='fits',
        dual_segmenter=camera, trend_continuation_fallback=enabled, allow_adapt=False)
    known = []
    for index, command in enumerate(nominal[:4]):
        region = command.region.buffer(-index*.06, join_style=2)
        primary = box(-5, -5, 0, 5) if bad_view == 'classical' and index == 3 else region
        camera.frames = index+1
        camera.alternative_region = box(-5, -5, 0, 5) if bad_view == 'learned' and index == 3 else region
        estimate, _ = policy.observe(command, Layer(command.z, primary), Detection('shift', 1., {}), index)
        known.append(estimate)
    return policy, nominal, known


def test_damping_only_continues_unchanged_commands_and_preserves_observations():
    control, nominal, known = observed_policy(False)
    context = DecisionContext(0)
    detection = Detection('shift', 1., {})
    assert control.decide(known, nominal, detection, context=context).action == 'scrap'
    policy, nominal, known = observed_policy()
    states = [policy.state, policy.alternative.state]
    before = [(list(s.residuals), [l.region.wkb for l in s.observations],
               s.uncertainty_mm, s.trend_shrinkage_queries) for s in states]
    decision = policy.decide(known, nominal, detection, context=context)
    assert decision.action == 'continue' and decision.verdict.ok
    assert decision.plan.layers == nominal[len(known):]
    assert decision.considered[-1]['both_camera_forecasts_pass']
    assert decision.considered[-1]['command_change_fraction'] == 0
    assert policy.damped_continuation_checks == 2 and policy.damped_continuation_queries > 0
    for state, (residuals, shapes, uncertainty, queries) in zip(states, before):
        assert list(state.residuals) == residuals
        assert [l.region.wkb for l in state.observations] == shapes
        assert state.uncertainty_mm == uncertainty and state.trend_shrinkage_queries == queries
        assert not state.shrink_trend


@pytest.mark.parametrize('view', ['classical', 'learned'])
def test_damping_cannot_repair_an_already_failing_prefix_in_either_view(view):
    policy, nominal, known = observed_policy(bad_view=view)
    decision = policy.decide(known, nominal, Detection('shift', 1., {}), context=DecisionContext(0))
    assert decision.action == 'scrap'
    assert not decision.considered[-1]['both_camera_forecasts_pass']


@pytest.mark.parametrize('name', ['continue', 'compensate:blend over 30 layers'])
def test_existing_passing_plan_prevents_any_damped_evaluation(monkeypatch, name):
    policy, nominal, known = observed_policy()
    original = [dict(name=name, plan=Plan(name, nominal[4:], 1.),
                     verdict=Verdict(True), executable=True)]
    monkeypatch.setattr(policy, '_camera_candidates', lambda *args: original)
    assert policy.candidates(known, nominal) is original
    assert policy.damped_continuation_checks == policy.damped_continuation_queries == 0


def test_damped_continuation_obeys_execution_validation():
    policy, nominal, known = observed_policy()
    def reject(layers):
        raise ValueError('printer cannot execute remaining path')
    decision = policy.decide(known, nominal, Detection('shift', 1., {}),
                             context=DecisionContext(0, reject))
    assert decision.action == 'scrap'
    assert not decision.considered[-1]['both_camera_forecasts_pass']
    assert not decision.considered[-1]['other_camera_executable']


def test_fit_domain_never_receives_damped_continuation(monkeypatch):
    policy, nominal, known = observed_policy()
    policy.spec = {'holes': [{'center': [0, 0]}]}
    policy.forecast_source = 'trained'
    original = [dict(name='continue', verdict=Verdict(False), executable=True)]
    monkeypatch.setattr(policy, '_camera_candidates', lambda *args: original)
    assert policy.candidates(known, nominal) is original
    assert policy.damped_continuation_checks == 0


def test_global_damping_and_fallback_cannot_be_combined():
    with pytest.raises(ValueError, match='unchanged original'):
        FallbackNeuralPolicy({}, [], .25, checkpoint='unused', forecast_scope='fits',
            shrink_trend=True, trend_continuation_fallback=True)


@pytest.mark.parametrize('args', [
    ['--boundary-trend-continuation-fallback'],
    ['--boundary-trend-continuation-fallback', '--boundary-fallback', '--contour-trend-shrinkage'],
])
def test_cli_rejects_incompatible_fallback_configuration(args, capsys):
    from recovery.eval.mujoco_eval import main
    with pytest.raises(SystemExit) as exc:
        main(['--methods', 'contour', *args])
    assert exc.value.code == 2
    assert 'trend continuation fallback requires' in capsys.readouterr().err


def test_audit_detects_displaced_recovery_unpaired_views_or_changed_commands():
    policy, nominal, known = observed_policy()
    decision = policy.decide(known, nominal, Detection('shift', 1., {}), context=DecisionContext(0))
    row = dict(has_fit_constraints=False, damped_continuation_checks=policy.damped_continuation_checks,
        damped_continuation_queries=policy.damped_continuation_queries,
        actions=[dict(action=decision.action, considered=decision.considered)])
    validate_damped_continuation(row, True)
    with pytest.raises(AssertionError, match='declared domain'):
        validate_damped_continuation(row, False)
    for index, change, message in [
        (0, {'ok': True}, 'passing original'),
        (-1, {'both_camera_forecasts_pass': False}, 'views disagree'),
        (-1, {'command_change_fraction': .01}, 'changed commands'),
        (-1, {'other_camera_executable': False}, 'not executable'),
    ]:
        broken = deepcopy(row)
        broken['actions'][0]['considered'][index].update(change)
        with pytest.raises(AssertionError, match=message):
            validate_damped_continuation(broken, True)
    with pytest.raises(AssertionError, match='missing paired'):
        validate_damped_continuation(dict(row, damped_continuation_checks=1), True)
