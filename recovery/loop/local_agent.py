"""Local language-model supervision over executable, spec-checked candidates.

No simulator access, API key, arbitrary code execution or silent model fallback.
Every actual request and validated decision is retained in the policy audit.
"""
import hashlib
import json
import time
import urllib.parse
import urllib.request

from .decide import Decision, DecisionContext
from .neural import NeuralPolicy


class LocalSupervisor:
    def __init__(self, endpoint='http://127.0.0.1:8089', model='local-qwen3', timeout=60):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('local supervisor must use a localhost HTTP server')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('unexpected credentials or query in local endpoint')
        self.endpoint, self.model, self.timeout = endpoint.rstrip('/'), model, timeout
        self.audit = []

    def choose(self, observation, candidates):
        ids = [c['id'] for c in candidates if c['ok'] and c['executable']]
        if not ids:
            raise ValueError('no allowed agent action')
        schema = dict(type='object', properties=dict(candidate=dict(type='integer', enum=ids),
                      reason=dict(type='string')), required=['candidate', 'reason'], additionalProperties=False)
        system = ('You supervise recovery of a 3D print. Choose one candidate using the supplied tool results. '
            'Only candidates marked ok and executable are allowed. Aim for a functional completed print. '
            'Prefer continuing when it passes; unnecessary intervention consumes a limited budget. '
            'For recovery, consider functional margin, robustness under uncertainty, material change, '
            'and remaining correction distance. Larger margins are better. Negative robust margin means '
            'the uncertainty envelope fails, even if the central prediction passes. '
            'Do not invent measurements or change the functional specification. Reply with JSON and a short reason.')
        evidence = dict(observation=observation, candidates=candidates)
        payload = dict(model=self.model, messages=[dict(role='system', content=system),
            dict(role='user', content=json.dumps(evidence, allow_nan=False, separators=(',', ':')))],
            temperature=0., seed=2026, max_tokens=160,
            chat_template_kwargs=dict(enable_thinking=False),
            response_format=dict(type='json_schema', json_schema=dict(name='recovery_decision', strict=True, schema=schema)))
        record = dict(request=payload, accepted=False)
        self.audit.append(record)
        started = time.monotonic()
        try:
            request = urllib.request.Request(self.endpoint + '/v1/chat/completions',
                data=json.dumps(payload, allow_nan=False).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            record['response'] = result
            content = result['choices'][0]['message']['content']
            selected = json.loads(content)
            if (set(selected) != {'candidate', 'reason'} or type(selected['candidate']) is not int
                    or selected['candidate'] not in ids or not isinstance(selected['reason'], str)):
                raise ValueError('local model selected an invalid or unverified action')
            if result['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('local model response did not finish normally')
            record.update(accepted=True, selected=selected)
            return selected
        except Exception as exc:
            record['error'] = repr(exc)
            raise RuntimeError('local supervisor failed; refusing to count an unsupervised decision') from exc
        finally:
            record['seconds'] = time.monotonic() - started
            record['request_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class AgentNeuralPolicy(NeuralPolicy):
    def __init__(self, *args, supervisor=None, review_interval_layers=8, **kwargs):
        super().__init__(*args, **kwargs)
        if review_interval_layers < 1:
            raise ValueError('supervisor review interval must be positive')
        self.supervisor = supervisor or LocalSupervisor()
        self.review_interval_layers = review_interval_layers
        self.last_continue_review = None

    def decide(self, known, current_plan, detection, *, context=None):
        if detection.label == 'detach':
            return Decision('scrap', 'part detached: nothing to build on')
        context = context or DecisionContext()
        candidates = self.candidates(known, current_plan, context)
        considered = [{k: v for k, v in c.items() if k not in ('plan', 'predicted', 'verdict', 'stress')}
            | dict(id=i, ok=c['verdict'].ok, violations=c['verdict'].violations,
                   fidelity=c['plan'].fidelity, adapts_remaining=context.adapts_remaining)
            for i, c in enumerate(candidates)]
        if not any(c['ok'] and c['executable'] for c in considered):
            return Decision('scrap', 'no executable recovery meets the functional spec',
                            None, candidates[0]['verdict'], considered)
        # A passing continuation is the sole allowed action here. Rechecking
        # its geometry still happens on EVERY warning. Reuse a recent model
        # review only for that unambiguous action, never for an adaptation or
        # when any fresh check fails. Cached reviews are not model requests.
        only_continue = len(candidates) == 1 and candidates[0]['name'] == 'continue'
        if only_continue and self.last_continue_review:
            previous_layer, request_hash = self.last_continue_review
            if len(known) - previous_layer < self.review_interval_layers:
                chosen = candidates[0]
                considered.append(dict(agent_review_reused=request_hash,
                                       fresh_function_checks_passed=True, reviewed_at_layer=previous_layer))
                return Decision('continue', 'fresh checks pass; recent local-agent continuation review retained',
                                chosen['plan'], chosen['verdict'], considered)
        observation = dict(printed_layers=len(known), total_layers=len(self.nominal),
            detector_label=detection.label, deviation_mm=detection.deviation_mm,
            uncertainty_mm=self.state.uncertainty_mm, adapts_remaining=context.adapts_remaining,
            checkpoint_sha256=self.state.model.sha256)
        selected = self.supervisor.choose(observation, considered)
        chosen = candidates[selected['candidate']]
        self.last_continue_review = ((len(known), self.supervisor.audit[-1]['request_sha256'])
                                     if chosen['name'] == 'continue' else None)
        considered.append(dict(agent=self.supervisor.audit[-1]))
        return Decision('continue' if chosen['name'] == 'continue' else 'adapt',
            'local Qwen supervisor: ' + selected['reason'], chosen['plan'], chosen['verdict'], considered)
