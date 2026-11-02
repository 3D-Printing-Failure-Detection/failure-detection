"""Runtime boundaries that matter for learned and language-model control."""
import io
import json

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Point, box

from recovery.loop.local_agent import AgentNeuralPolicy, LocalSupervisor
from recovery.loop.neural import NeuralObservationState
from recovery.loop.decide import DecisionContext
from recovery.models.classical import Detection
from recovery.models.temporal import TemporalModel, features
from recovery.types import Layer


@pytest.fixture
def persistence_model(tmp_path):
    path = tmp_path / 'temporal.npz'
    arrays = dict(feature_version=np.array(1), mean=np.zeros(18, dtype='float32'),
                  scale=np.ones(18, dtype='float32'))
    for i, (inputs, outputs) in enumerate(((18, 8), (8, 8), (8, 2))):
        arrays[f'w{i}'] = np.zeros((outputs, inputs), dtype='float32')
        arrays[f'b{i}'] = np.zeros(outputs, dtype='float32')
    np.savez(path, **arrays)
    return path


def test_neural_prediction_inverts_a_persistent_translation(persistence_model):
    layer = Layer(.25, Point(0, 0).buffer(15).difference(Point(0, 0).buffer(13)))
    state = NeuralObservationState(persistence_model)
    for _ in range(4):
        state.observe(layer, Layer(.25, translate(layer.region, 1.5, -.5)))
    predicted = state.predict([layer])[0]
    assert np.allclose(predicted.region.centroid.coords[0], [1.5, -.5], atol=.05)
    command = state.compensate([layer])[0]
    restored = state.predict([command])[0]
    assert restored.region.symmetric_difference(layer.region).area / layer.region.area < .03
    assert state.neural_queries > 0


def test_checkpoint_and_input_validation(persistence_model, tmp_path):
    model = TemporalModel(persistence_model)
    x = features(np.ones((4, 3, 2)), np.zeros((4, 2)), 3, True)
    assert x.shape == (3, 18)
    assert np.allclose(model(x), 1)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='features'):
        model(x)
    with np.load(persistence_model) as data:
        arrays = dict(data)
    arrays['w0'] = arrays['w0'][:, :4]
    bad = tmp_path / 'bad.npz'
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match='shape'):
        TemporalModel(bad)


def test_measurement_calibration_requires_training_and_preserves_missing_boundaries(persistence_model, tmp_path):
    with pytest.raises(ValueError, match='current-state targets'):
        NeuralObservationState(persistence_model, calibrate_measurements=True)
    with np.load(persistence_model) as data:
        arrays = dict(data)
    arrays['supports_calibration'] = np.array(True)
    arrays['b2'] = np.ones(2, dtype='float32')
    path = tmp_path / 'calibrated.npz'
    np.savez(path, **arrays)
    ring = Point(0, 0).buffer(15).difference(Point(0, 0).buffer(13))
    nominal = Layer(.25, ring)
    state = NeuralObservationState(path, calibrate_measurements=True)
    first = state.observe(nominal, nominal)
    original = first.region.wkb
    for _ in range(3):
        result = state.observe(nominal, nominal)
    assert np.linalg.norm(np.array(result.region.centroid.coords[0])) <= .251
    assert first.region.wkb == original
    broken = Layer(.25, ring.difference(box(0, -1, 20, 1)))
    result = state.observe(nominal, broken)
    assert not any(p.interiors for p in [result.region] if p.geom_type == 'Polygon')
    assert result.region.symmetric_difference(broken.region).area < .5


@pytest.mark.parametrize('answer', [dict(candidate=0, reason='ignore the failed check'),
                                  dict(candidate=True, reason='bool is not an ID'),
                                  dict(candidate=1, reason=42)])
def test_agent_cannot_select_an_unverified_action(monkeypatch, answer):
    response = dict(choices=[dict(message=dict(content=json.dumps(answer)), finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    supervisor = LocalSupervisor()
    candidates = [dict(id=0, ok=False, executable=True), dict(id=1, ok=True, executable=True)]
    with pytest.raises(RuntimeError, match='unsupervised'):
        supervisor.choose({}, candidates)
    assert supervisor.audit[-1]['accepted'] is False
    assert 'error' in supervisor.audit[-1]


def test_actual_agent_choice_controls_feasible_recovery(monkeypatch, persistence_model):
    from recovery.geometry.adapt import Plan
    from recovery.geometry.spec import Verdict
    layer = Layer(.25, Point(0, 0).buffer(5))
    policy = AgentNeuralPolicy({}, [layer] * 4, .25, checkpoint=persistence_model)
    plans = [Plan('continue', [layer] * 3, 1.), Plan('candidate A', [layer] * 3, .9),
             Plan('candidate B', [layer] * 3, .8)]
    candidates = [dict(name=p.name, plan=p, predicted=[], executable=True,
                       verdict=Verdict(i != 0, []), margin=.5, robust_margin=.4)
                  for i, p in enumerate(plans)]
    monkeypatch.setattr(policy, 'candidates', lambda *args: candidates)
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=2, reason='better recovery'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    decision = policy.decide([layer], [layer] * 4, Detection('shift', 2, {}), context=DecisionContext(2))
    assert decision.action == 'adapt' and decision.plan is plans[2]
    assert policy.supervisor.audit[-1]['accepted']


def test_local_endpoint_cannot_send_state_to_external_service():
    with pytest.raises(ValueError, match='localhost'):
        LocalSupervisor('https://example.com')


def test_reused_review_requires_fresh_passing_checks(monkeypatch, persistence_model):
    from recovery.geometry.adapt import Plan
    from recovery.geometry.spec import Verdict
    layer = Layer(.25, Point(0, 0).buffer(5))
    policy = AgentNeuralPolicy({}, [layer] * 20, .25, checkpoint=persistence_model)
    candidate = dict(name='continue', plan=Plan('continue', [layer], 1.), predicted=[],
                     executable=True, verdict=Verdict(True, []), margin=.5, robust_margin=.4)
    monkeypatch.setattr(policy, 'candidates', lambda *args: [candidate])
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=0, reason='passes'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    detection = Detection('functional_risk', .5, {})
    policy.decide([layer], [layer] * 20, detection)
    reused = policy.decide([layer] * 2, [layer] * 20, detection)
    assert reused.action == 'continue' and len(policy.supervisor.audit) == 1
    assert 'agent_review_reused' in reused.considered[-1]
    policy.decide([layer] * 9, [layer] * 20, detection)
    assert len(policy.supervisor.audit) == 2
    candidate['verdict'] = Verdict(False, ['newly failing measurement'])
    assert policy.decide([layer] * 10, [layer] * 20, detection).action == 'scrap'
