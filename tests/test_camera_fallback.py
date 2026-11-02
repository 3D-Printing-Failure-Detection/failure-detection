from types import SimpleNamespace

import pytest
from shapely.geometry import box

from recovery.loop.camera_fallback import FallbackNeuralPolicy
from recovery.models.classical import Detection
from recovery.types import Layer


def setup_policy():
    layers = [Layer(z, box(-5, -5, 5, 5)) for z in (.25, .5, .75)]
    spec = {'present': [{'z': [0., 1.], 'min_area_frac': .8}]}
    camera = SimpleNamespace(frames=1, alternative_region=layers[0].region)
    policy = FallbackNeuralPolicy(spec, layers, .25, checkpoint='unused-for-nonfits',
                                 forecast_scope='fits', dual_segmenter=camera)
    return layers, camera, policy


def test_passing_primary_plan_is_preserved_despite_disagreeing_camera():
    layers, camera, policy = setup_policy()
    camera.alternative_region = box(-5, -5, 0, 5)
    observed, detection = policy.observe(layers[0], layers[0], Detection('shift', 2., {}), 0)
    decision = policy.decide([observed], layers, detection)
    assert decision.action == 'continue'
    assert all(c['observation_model'] == 'classical camera' for c in decision.considered)
    assert len(policy.alternative.state.observations) == 1
    assert policy.alternative.state.observations[0].region.area < observed.region.area


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
