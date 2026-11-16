"""Runtime boundaries that matter for learned and language-model control."""
import io
import json

import numpy as np
import pytest
from shapely.affinity import translate
from shapely.geometry import Point, box

from recovery.loop.local_agent import AgentNeuralPolicy, LocalSupervisor, rank_for_review
from recovery.loop.neural import NeuralObservationState, NeuralPolicy
from recovery.loop.contour import ContourPolicy
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


@pytest.mark.parametrize('part_name,source', [('cup', 'analytical'), ('spacer', 'trained')])
def test_fit_scope_preserves_the_container_forecast(persistence_model, part_name, source):
    from recovery.parts import get_part
    part = get_part(part_name)
    layers = part.layers()
    policy = NeuralPolicy(part.spec, layers, part.layer_h, checkpoint=persistence_model, forecast_scope='fits')
    baseline = ContourPolicy(part.spec, layers, part.layer_h)
    for layer in layers[:4]:
        measured = Layer(layer.z, translate(layer.region, .5, -.25))
        policy.state.observe(layer, measured)
        baseline.state.observe(layer, measured)
    actual = policy.state.predict(layers[4:5])[0]
    expected = baseline.state.predict(layers[4:5])[0]
    assert policy.forecast_source == source
    if source == 'analytical':
        assert actual.region.wkb == expected.region.wkb
        assert getattr(policy.state, 'neural_queries', 0) == 0
    else:
        assert policy.state.neural_queries > 0


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


@pytest.mark.parametrize('camera_scope,forecast_scope', [('all', 'fits'), ('nonfits', 'all')])
def test_evaluator_rejects_mismatched_measurement_domains(persistence_model, tmp_path, camera_scope, forecast_scope):
    from recovery.eval.mujoco_eval import main
    boundary = tmp_path/'boundary.npz'
    boundary.write_bytes(b'different camera identity')
    with pytest.raises(SystemExit) as error:
        main(['--methods', 'neural_risk', '--neural-checkpoint', str(persistence_model),
              '--boundary-checkpoint', str(boundary), '--boundary-scope', camera_scope,
              '--neural-forecast-scope', forecast_scope, '--out', str(tmp_path/'results')])
    assert error.value.code == 2
    assert not (tmp_path/'results').exists()


def test_equivariant_forecast_rotates_with_observed_motion(persistence_model, tmp_path):
    with np.load(persistence_model) as data:
        arrays = dict(data)
    rng = np.random.default_rng(83)
    for key in list(arrays):
        if key[0] in ('w', 'b'):
            arrays[key] = rng.normal(0, .2, arrays[key].shape).astype('float32')
    arrays['equivariant'] = np.array(True)
    path = tmp_path/'equivariant.npz'
    np.savez(path, **arrays)
    model = TemporalModel(path)
    x = features(rng.normal(size=(4, 10, 2)), rng.normal(size=(4, 2)), 10, False)
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype='float32')
    rotated = x.copy()
    rotated[:, :16] = (x[:, :16].reshape(-1, 8, 2) @ rotation.T).reshape(-1, 16)
    assert np.allclose(model(rotated), model(x) @ rotation.T, atol=1e-5)
    x[:, :16] = 0
    assert np.allclose(model(x), 0)


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


def test_compact_agent_request_preserves_every_eligible_choice(monkeypatch):
    response = dict(choices=[dict(message=dict(content=json.dumps(
        dict(candidate=3, reason='choose the second eligible tradeoff'))), finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    supervisor = LocalSupervisor()
    candidates = [dict(id=0, ok=False, executable=True),
                  dict(id=1, ok=True, executable=True),
                  dict(id=2, ok=True, executable=True, review_eligible=False),
                  dict(id=3, ok=True, executable=True)]
    selected = supervisor.choose({}, candidates)
    request = supervisor.audit[-1]['request']
    sent = json.loads(request['messages'][1]['content'])['candidates']
    assert {c['id'] for c in sent} == {1, 3}
    assert set(request['response_format']['json_schema']['schema']['properties']['candidate']['enum']) == {1, 3}
    assert selected['candidate'] == 3
    assert len(candidates) == 4


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
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=2, reason='robust margin is 123 mm'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    decision = policy.decide([layer], [layer] * 4, Detection('shift', 2, {}), context=DecisionContext(2))
    assert decision.action == 'adapt' and decision.plan is plans[2]
    assert policy.supervisor.audit[-1]['accepted']
    assert 'candidate B' in decision.reason and '123' not in decision.reason
    assert policy.supervisor.audit[-1]['selected']['reason'] == 'robust margin is 123 mm'


def test_agent_rank_one_selects_the_best_plan_even_when_generated_last(monkeypatch, persistence_model):
    from recovery.geometry.adapt import Plan
    from recovery.geometry.spec import Verdict
    layer = Layer(.25, Point(0, 0).buffer(5))
    policy = AgentNeuralPolicy({}, [layer]*4, .25, checkpoint=persistence_model)
    plans = [Plan('continue', [layer]*3, 1.), Plan('weak correction', [layer]*3, .9),
             Plan('clearance reserve', [layer]*3, .8)]
    candidates = [dict(name=p.name, plan=p, predicted=[], executable=True,
                       verdict=Verdict(i != 0, []), margin=.5, robust_margin=i*.2)
                  for i, p in enumerate(plans)]
    monkeypatch.setattr(policy, 'candidates', lambda *args: candidates)
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=1, reason='rank one'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    decision = policy.decide([layer], [layer]*4, Detection('shift', 2, {}))
    assert decision.plan is plans[2]
    evidence = json.loads(policy.supervisor.audit[-1]['request']['messages'][1]['content'])
    first = evidence['candidates'][0]
    assert first['id'] == first['functional_rank'] == 1
    assert first['name'] == 'clearance reserve'


def test_local_endpoint_cannot_send_state_to_external_service():
    with pytest.raises(ValueError, match='localhost'):
        LocalSupervisor('https://example.com')


def test_agent_cannot_override_a_materially_better_verified_fit_option(monkeypatch):
    candidates = [dict(id=0, name='fragile', ok=True, executable=True, margin=.38, robust_margin=-1.,
                      fit_clearance_mm=.29, fit_robust_clearance_mm=-.17),
                  dict(id=1, name='clearance reserve', ok=True, executable=True, margin=.38, robust_margin=-1.,
                      fit_clearance_mm=.29, fit_robust_clearance_mm=-.05)]
    ranked = rank_for_review(candidates, .25)
    assert not ranked[0]['review_eligible'] and ranked[1]['review_eligible']
    assert ranked[1]['functional_rank'] == 1
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=0, reason='prefer fidelity'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(RuntimeError, match='unverified action'):
        LocalSupervisor().choose({}, ranked)


def test_review_preserves_real_fit_tradeoffs():
    candidates = [dict(id=0, ok=True, executable=True, margin=.5, robust_margin=-1.,
                      fit_clearance_mm=.5, fit_robust_clearance_mm=-.17),
                  dict(id=1, ok=True, executable=True, margin=.3, robust_margin=-1.,
                      fit_clearance_mm=.29, fit_robust_clearance_mm=-.05)]
    assert all(c['review_eligible'] for c in rank_for_review(candidates, .25))


def test_reused_review_requires_fresh_passing_checks(monkeypatch, persistence_model):
    from recovery.geometry.adapt import Plan
    from recovery.geometry.spec import Verdict
    layer = Layer(.25, Point(0, 0).buffer(5))
    policy = AgentNeuralPolicy({}, [layer] * 20, .25, checkpoint=persistence_model)
    candidate = dict(name='continue', plan=Plan('continue', [layer], 1.), predicted=[],
                     executable=True, verdict=Verdict(True, []), margin=.5, robust_margin=.4)
    monkeypatch.setattr(policy, 'candidates', lambda *args: [candidate])
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=1, reason='passes'))),
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


class _FakeClaude:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.messages = text, error, self
    def create(self, **kw):
        if self.error:
            raise self.error
        from types import SimpleNamespace as N
        return N(content=[N(type='text', text=self.text)], stop_reason='end_turn',
                 usage=N(input_tokens=10, output_tokens=5))


def _qwen_ok(monkeypatch, candidate):
    response = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate=candidate, reason='qwen'))),
                                  finish_reason='stop')])
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(response).encode()))


CANDS = [dict(id=1, ok=True, executable=True), dict(id=2, ok=True, executable=True)]


def test_claude_supervisor_answers_in_audit_compatible_shape():
    from recovery.loop.local_agent import ClaudeSupervisor
    sup = ClaudeSupervisor(client=_FakeClaude('```json\n{"candidate": 2, "reason": "ok"}\n```'))
    assert sup.choose({}, CANDS)['candidate'] == 2
    rec = sup.audit[-1]
    assert rec['accepted'] and rec['response']['choices'][0]['finish_reason'] == 'stop'


def test_claude_failure_falls_back_to_qwen_visibly(monkeypatch):
    from recovery.loop.local_agent import ClaudeSupervisor, FallbackSupervisor
    _qwen_ok(monkeypatch, 1)
    sup = FallbackSupervisor(ClaudeSupervisor(client=_FakeClaude(error=TimeoutError('down'))), LocalSupervisor())
    assert sup.choose({}, CANDS)['candidate'] == 1
    assert [a['accepted'] for a in sup.audit] == [False, True]
    assert 'TimeoutError' in sup.audit[-1]['fallback_for']['error']


def test_claude_invalid_choice_falls_back_and_claude_success_skips_qwen(monkeypatch):
    from recovery.loop.local_agent import ClaudeSupervisor, FallbackSupervisor
    _qwen_ok(monkeypatch, 2)
    bad = FallbackSupervisor(ClaudeSupervisor(client=_FakeClaude('{"candidate": 9, "reason": "x"}')), LocalSupervisor())
    assert bad.choose({}, CANDS)['candidate'] == 2 and len(bad.audit) == 2
    good = FallbackSupervisor(ClaudeSupervisor(client=_FakeClaude('{"candidate": 1, "reason": "x"}')), LocalSupervisor())
    assert good.choose({}, CANDS)['candidate'] == 1 and len(good.audit) == 1
