from copy import deepcopy

import pytest

from tools.audit_neural_experiment import validate_outcomes, agent_decision_opportunities


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
