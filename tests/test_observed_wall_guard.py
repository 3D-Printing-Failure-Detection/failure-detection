from copy import deepcopy
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from recovery.geometry.adapt import Plan
from recovery.geometry.spec import Verdict, check
from recovery.loop.camera_fallback import FallbackNeuralPolicy
from recovery.models.classical import Detection
from recovery.parts import get_part
from recovery.types import Layer
from tools.audit_neural_experiment import validate_observed_wall_guard


def wall_policy(width, enabled=True, adapted_command=False):
    ring = lambda width: box(-5, -5, 5, 5).difference(box(-5+width, -5+width, 5-width, 5-width))
    nominal = [Layer((i+1)*.25, ring(2.)) for i in range(3)]
    spec = dict(container=dict(base_h=0, capacity_mm3=[0, 1000]), min_wall_mm=1.6)
    learned = Layer(.25, ring(width))
    camera = SimpleNamespace(frames=1, alternative_region=learned.region)
    policy = FallbackNeuralPolicy(spec, nominal, .25, checkpoint='unused-for-nonfits',
        forecast_scope='fits', dual_segmenter=camera, observed_wall_guard=enabled,
        joint_camera=True, trend_continuation_fallback=True, allow_adapt=False)
    command = learned if adapted_command else nominal[0]
    observed, detection = policy.observe(command, nominal[0], Detection('ok', 0., {'source': 'camera'}), 0)
    return policy, nominal, learned, observed, detection


def test_observed_hard_wall_overrides_passing_primary_without_rewriting_history():
    policy, nominal, learned, observed, detection = wall_policy(.6)
    assert not check([learned, *nominal[1:]], nominal, policy.spec, .25).ok
    assert detection.label == 'functional_risk' and detection.info['source'] == 'camera'
    before = [[layer.region.wkb for layer in owner.state.observations] for owner in [policy, policy.alternative]]
    decision = policy.decide([observed], nominal, detection)
    assert decision.action == 'scrap'
    assert all(not c['ok'] and c['observed_wall_guard']['layer'] == 0 for c in decision.considered)
    assert policy.wall_guard_checks == 1 and len(policy.wall_guard_findings) == 1
    assert policy.damped_continuation_checks == 0
    assert before == [[layer.region.wkb for layer in owner.state.observations] for owner in [policy, policy.alternative]]


def test_wall_guard_uses_original_functional_design_after_a_command_change():
    policy, nominal, learned, observed, detection = wall_policy(.6, adapted_command=True)
    assert policy.wall_guard_findings  # Comparing with the thin command would hide the defect.
    assert policy.decide([observed], nominal, detection).action == 'scrap'


@pytest.mark.parametrize('width', [1.4, 2.])
def test_soft_thickness_or_good_wall_does_not_override_passing_primary(width):
    policy, nominal, learned, observed, detection = wall_policy(width)
    assert not policy.wall_guard_findings
    assert policy.decide([observed], nominal, detection).action == 'continue'
    assert policy.damped_continuation_checks == 0


def test_disabled_guard_preserves_the_previous_disagreement_behavior():
    policy, nominal, learned, observed, detection = wall_policy(.6, enabled=False)
    assert policy.wall_guard_checks == 0 and not policy.wall_guard_findings
    assert policy.decide([observed], nominal, detection).action == 'continue'


def test_damped_continuation_cannot_resurrect_a_guarded_prefix(monkeypatch):
    policy, nominal, learned, observed, detection = wall_policy(.6)
    original = [dict(name='continue', plan=Plan('continue', nominal[1:], 1.),
                     verdict=Verdict(False, ['forecast fails']), executable=True)]
    monkeypatch.setattr(policy, '_camera_candidates', lambda *args: original)
    candidates = policy.candidates([observed], nominal)
    assert len(candidates) == 1 and not candidates[0]['verdict'].ok
    assert 'observed_wall_guard' in candidates[0] and policy.damped_continuation_checks == 0
    assert original[0]['verdict'].violations == ['forecast fails']


def guard_row():
    part = get_part('mug')
    index = 101
    finding = dict(layer=index, thickness_mm=.6 * part.spec['min_wall_mm'], hard_thin_excess=.04)
    candidate = dict(name='continue', ok=False, observed_wall_guard=finding)
    return dict(part='mug', has_fit_constraints=False, frames=index+1, layers_printed=index+1,
                outcome='scrapped', wall_guard_findings=[finding],
                wall_guard_checks=sum(l.z > part.spec['container']['base_h'] for l in part.layers()[:index+1]),
                actions=[dict(layer=index, action='scrap', considered=[candidate])])


def test_guard_audit_rejects_bypassed_missing_or_future_evidence():
    row = guard_row()
    validate_observed_wall_guard(row, True)
    for change, message in [
        (lambda r: r.update(wall_guard_checks=0), 'every eligible'),
        (lambda r: r['wall_guard_findings'][0].update(layer=102), 'unobserved layer'),
        (lambda r: r['wall_guard_findings'][0].update(hard_thin_excess=.02), 'hard-thin observation'),
        (lambda r: r['actions'][0]['considered'][0].update(ok=True), 'eligible or damped'),
        (lambda r: r['actions'][0]['considered'].append(dict(name='correction', ok=True)), 'unguarded plan'),
        (lambda r: r['actions'][0].update(considered=[]), 'reach the decision'),
        (lambda r: r['actions'][0].update(action='continue'), 'did not stop'),
    ]:
        broken = deepcopy(row)
        change(broken)
        with pytest.raises(AssertionError, match=message):
            validate_observed_wall_guard(broken, True)
    with pytest.raises(AssertionError, match='every eligible'):
        validate_observed_wall_guard(row, False)


def test_cli_wall_guard_requires_the_existing_dual_camera_path(capsys):
    from recovery.eval.mujoco_eval import main
    with pytest.raises(SystemExit) as exc:
        main(['--methods', 'contour', '--boundary-observed-wall-guard'])
    assert exc.value.code == 2
    assert 'requires --boundary-fallback' in capsys.readouterr().err
