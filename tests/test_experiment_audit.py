from copy import deepcopy

import pytest

from tools.audit_neural_experiment import (validate_outcomes, agent_decision_opportunities,
                                           validate_base_camera_counts, validate_model_provenance,
                                           validate_fallback_uncertainty)


def paired_rows():
    row = dict(part='cup', kind='shift', seed=70, profile='cartesian', method='contour',
               k0=40, severity=.4, n_layers=160, layers_printed=160,
               clear_failure_on_adapt=False, outcome='finished', in_place=True,
               scrapped=False, defective_finished=False, violations=[])
    other = dict(row, profile='corexy', method='neural_risk_agent')
    return [row, other]


def test_accepts_paired_completions_and_honest_scraps():
    rows = paired_rows()
    rows[1].update(outcome='scrapped', in_place=False, scrapped=True,
                   layers_printed=42, violations=['no feasible continuation'])
    validate_outcomes(rows)


@pytest.mark.parametrize('change, message', [
    ({'layers_printed': 159}, 'incomplete print'),
    ({'outcome': 'scrapped'}, 'scrap classification'),
    ({'violations': ['hole too small']}, 'final violations'),
    ({'clear_failure_on_adapt': True}, 'failure persistence'),
    ({'k0': 41}, 'unpaired failure conditions'),
    ({'severity': .5}, 'unpaired failure conditions'),
    ({'n_layers': 161, 'layers_printed': 161}, 'unpaired failure conditions'),
])
def test_rejects_corrupted_outcomes_or_pairing(change, message):
    rows = deepcopy(paired_rows())
    rows[1].update(change)
    with pytest.raises(AssertionError, match=message):
        validate_outcomes(rows)


def test_clean_control_cannot_hide_an_injected_failure():
    rows = paired_rows()
    for row in rows:
        row['kind'] = 'clean'
    with pytest.raises(AssertionError, match='clean control'):
        validate_outcomes(rows)


def test_agent_unreached_when_no_eligible_correction_exists():
    rows = [dict(method='neural_risk_agent', actions=[dict(action='scrap', considered=[
        dict(id='continue', ok=False, executable=True),
        dict(id='correction', ok=True, executable=False)])])]
    assert agent_decision_opportunities(rows) == 0


def test_zero_calls_cannot_hide_a_bypassed_eligible_agent_decision():
    rows = [dict(method='neural_risk_agent', actions=[dict(action='adapt', considered=[
        dict(id='correction', ok=True, executable=True)])])]
    with pytest.raises(AssertionError, match='bypassed the agent'):
        agent_decision_opportunities(rows)
    rows[0]['actions'][0]['considered'].append(dict(agent={'accepted': True}))
    assert agent_decision_opportunities(rows) == 1


def test_agent_cannot_adapt_without_any_recorded_candidate_or_review():
    rows = [dict(method='neural_risk_agent', actions=[dict(action='adapt', considered=[])])]
    with pytest.raises(AssertionError, match='adaptation bypassed the agent'):
        agent_decision_opportunities(rows)


def test_base_camera_audit_accounts_for_each_observed_frame():
    row = dict(n_layers=6, frames=4, alternative_classical_base_frames=2,
               alternative_learned_frames=2)
    validate_base_camera_counts(row, True, [0, 1])
    with pytest.raises(AssertionError, match='base-camera'):
        validate_base_camera_counts(row, True, [0, 1, 2])
    row.update(frames=1, alternative_classical_base_frames=1, alternative_learned_frames=0)
    validate_base_camera_counts(row, True, [0, 1])
    row.update(alternative_classical_base_frames=0)
    validate_base_camera_counts(row, False, [0, 1])


def analytical_control():
    manifest = dict(instrumentation_version=4, arguments=dict(methods=['contour_risk'],
        neural_checkpoint=None, boundary_checkpoint=None, supervisor_manifest=None))
    row = dict(neural_queries=0, boundary_queries=0, agent_requests=0, agent_accepted=0,
        alternative_observations=0, measurement_source='classical',
        alternative_measurement_source=None, forecast_source='analytical')
    return manifest, [row]


def test_audit_rejects_missing_or_bypassed_fallback_uncertainty_guard():
    candidate = dict(name='correction',fallback_after_no_primary_plan=True,robust_margin=-1.,ok=False,
        fallback_uncertainty_required=True,fallback_uncertainty_passes=False)
    validate_fallback_uncertainty([candidate], True)
    for change, message in [({'ok':True}, 'remains eligible'),
                            ({'fallback_uncertainty_required':False}, 'not recorded'),
                            ({'fallback_uncertainty_passes':True}, 'margin differs')]:
        with pytest.raises(AssertionError, match=message):
            validate_fallback_uncertainty([dict(candidate,**change)], True)
    with pytest.raises(AssertionError, match='not recorded'):
        validate_fallback_uncertainty([candidate], False)
    validate_fallback_uncertainty([dict(candidate,robust_margin=0.,ok=True,fallback_uncertainty_passes=True)], True)
    validate_fallback_uncertainty([dict(name='continue',fallback_after_no_primary_plan=True,robust_margin=-1.,ok=True)], True)
    validate_fallback_uncertainty([dict(name='joint correction',both_camera_forecasts_pass=True,robust_margin=-1.,ok=True)], True)


def test_analytical_control_requires_explicit_mode_and_zero_model_use():
    manifest, rows = analytical_control()
    with pytest.raises(AssertionError, match='no trained model provenance'):
        validate_model_provenance(manifest, rows)
    assert validate_model_provenance(manifest, rows, analytical_only=True) == {}
    for key in ('neural_queries', 'boundary_queries', 'agent_requests', 'agent_accepted', 'alternative_observations'):
        changed = deepcopy(rows)
        changed[0][key] = 1
        with pytest.raises(AssertionError, match='model calls'):
            validate_model_provenance(manifest, changed, analytical_only=True)
    for key, value in [('measurement_source', 'learned'), ('alternative_measurement_source', 'checkpoint'),
                       ('forecast_source', 'trained')]:
        changed = deepcopy(rows)
        changed[0][key] = value
        with pytest.raises(AssertionError, match='learned'):
            validate_model_provenance(manifest, changed, analytical_only=True)


def test_analytical_mode_cannot_waive_neural_policy_or_asset_provenance():
    manifest, rows = analytical_control()
    for method in ('neural_risk', 'neural_risk_agent', 'neural_camera', 'unknown'):
        changed = deepcopy(manifest)
        changed['arguments']['methods'] = [method]
        with pytest.raises(AssertionError, match='model policy'):
            validate_model_provenance(changed, rows, analytical_only=True)
    for key in ('neural_checkpoint', 'boundary_checkpoint', 'supervisor_manifest'):
        changed = deepcopy(manifest)
        changed['arguments'][key] = 'model-asset'
        with pytest.raises(AssertionError, match='model assets'):
            validate_model_provenance(changed, rows, analytical_only=True)
    changed = dict(manifest, neural_checkpoint_sha256='pinned', final_neural_checkpoint_sha256='pinned')
    with pytest.raises(AssertionError, match='model provenance'):
        validate_model_provenance(changed, rows, analytical_only=True)
    assert validate_model_provenance(changed, rows) == {'neural': 'pinned'}
