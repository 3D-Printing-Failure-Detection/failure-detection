from types import SimpleNamespace

import pytest
from shapely.geometry import box

from recovery.loop.camera_fallback import FallbackNeuralPolicy
from recovery.loop.neural import NeuralPolicy
from recovery.loop.decide import DecisionContext
from recovery.geometry.adapt import Plan
from recovery.geometry.spec import Verdict
from recovery.models.classical import Detection
from recovery.types import Layer


def setup_policy():
    layers = [Layer(z, box(-5, -5, 5, 5)) for z in (.25, .5, .75)]
    spec = {'present': [{'z': [0., 1.], 'min_area_frac': .8}]}
    camera = SimpleNamespace(frames=1, alternative_region=layers[0].region)
    policy = FallbackNeuralPolicy(spec, layers, .25, checkpoint='unused-for-nonfits',
                                 forecast_scope='fits', dual_segmenter=camera)
    return layers, camera, policy


@pytest.mark.parametrize('joint_camera', [False, True])
def test_passing_primary_plan_is_preserved_despite_disagreeing_camera(joint_camera):
    layers, camera, policy = setup_policy()
    policy.joint_camera = joint_camera
    camera.alternative_region = box(-5, -5, 0, 5)
    observed, detection = policy.observe(layers[0], layers[0], Detection('shift', 2., {}), 0)
    decision = policy.decide([observed], layers, detection)
    assert decision.action == 'continue'
    assert all(c['observation_model'] == 'classical camera' for c in decision.considered)
    assert len(policy.alternative.state.observations) == 1
    assert policy.alternative.state.observations[0].region.area < observed.region.area
    assert policy.joint_camera_checks == 0


def test_learned_view_can_preserve_a_print_when_primary_has_no_feasible_plan():
    layers, camera, policy = setup_policy()
    measured = Layer(.25, box(-5, -5, 0, 5))
    observed, detection = policy.observe(layers[0], measured, Detection('collapse', 2., {}), 0)
    prefix_wkb = observed.region.wkb
    decision = policy.decide([observed], layers, detection)
    assert decision.action == 'continue' and decision.verdict.ok
    primary = [c for c in decision.considered if c['observation_model'] == 'classical camera']
    alternate = [c for c in decision.considered if c['observation_model'] == 'learned camera']
    assert primary and not any(c['ok'] for c in primary)
    assert any(c['ok'] and c['fallback_after_no_primary_plan'] for c in alternate)
    assert observed.region.wkb == prefix_wkb  # Neither measurement history is rewritten.
    assert policy.alternative.state.observations[0].region.area > observed.region.area


def test_alternative_measurement_must_be_from_the_current_frame():
    layers, camera, policy = setup_policy()
    camera.frames = 0
    with pytest.raises(ValueError, match='missing or stale'):
        policy.observe(layers[0], layers[0], Detection('ok', 0., {}), 0)


@pytest.mark.parametrize('joint_alternative', [False, True])
def test_cross_camera_recovery_needs_a_jointly_passing_alternative(monkeypatch, joint_alternative):
    layers, camera, policy = setup_policy()
    policy.joint_camera = True
    observed, _ = policy.observe(layers[0], layers[0], Detection('shift', 2., {}), 0)
    before = observed.region.wkb

    def candidate(name, ok=True, score=.8):
        plan = Plan(name, layers[1:], 1.)
        return dict(name=name, plan=plan, predicted=plan.layers, executable=True,
                    verdict=Verdict(ok, [] if ok else ['forecast fails'], {}),
                    margin=score, robust_margin=score, command_change_fraction=.01)

    primary = [candidate('continue', False), candidate('classical correction')]
    alternative = [candidate('continue', False), candidate('learned correction')]
    monkeypatch.setattr(NeuralPolicy, 'candidates', lambda *args: primary)
    monkeypatch.setattr(policy.alternative, 'candidates', lambda *args: alternative)
    monkeypatch.setattr(policy.alternative, '_evaluate', lambda known, plan, *args:
                        candidate(plan.name, False))
    monkeypatch.setattr(policy, '_evaluate', lambda known, plan, *args:
                        candidate(plan.name, joint_alternative, .3))
    candidates = policy.candidates([observed], layers, DecisionContext(3))
    decision = policy.pick_candidates(candidates, DecisionContext(3))
    assert policy.joint_camera_checks == 2
    assert observed.region.wkb == before
    assert primary[1]['verdict'].ok and alternative[1]['verdict'].ok
    if joint_alternative:
        assert decision.plan.name == 'learned correction'
        assert not candidates[1]['verdict'].ok
        assert candidates[-1]['both_camera_forecasts_pass']
        assert candidates[-1]['robust_margin'] == .3
    else:
        assert decision.plan.name == 'classical correction'
        assert len(candidates) == len(primary)
