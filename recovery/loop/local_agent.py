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
from .contour import ContourPolicy
from .neural import NeuralPolicy
from .risk_neural import RiskAwareNeuralPolicy
from .camera_fallback import FallbackNeuralPolicy, FallbackRiskNeuralPolicy


def rank_for_review(candidates, resolution_mm):
    """Tool-computed functional priorities; no simulator outcomes enter here.

    A local model must not trade a materially worse fit-risk score for changes
    smaller than measurement resolution in the central fit. Other genuine
    tradeoffs remain available for review. Scores are forecast scenarios, not
    calibrated success probabilities or physical safety guarantees.
    """
    candidates = [dict(c) for c in candidates]
    feasible = [c for c in candidates if c['ok'] and c['executable']]
    tolerance = resolution_mm/4
    for candidate in feasible:
        candidate['review_eligible'] = True
        if 'fit_robust_clearance_mm' not in candidate:
            continue
        for alternative in feasible:
            if ('fit_robust_clearance_mm' in alternative
                    and alternative['fit_robust_clearance_mm'] > candidate['fit_robust_clearance_mm'] + tolerance
                    and alternative.get('fit_clearance_mm', -1.) >= candidate.get('fit_clearance_mm', -1.) - tolerance
                    and alternative.get('robust_margin', -1.) >= candidate.get('robust_margin', -1.)
                    and alternative.get('margin', -1.) >= candidate.get('margin', -1.) - .02):
                candidate['review_eligible'] = False
                candidate['review_exclusion'] = 'functional-risk dominated by ' + alternative.get('name', str(alternative['id']))
                break
    ordered = sorted(feasible, key=lambda c: (c['review_eligible'],
        c.get('fit_robust_clearance_mm', -1.), c.get('fit_clearance_mm', -1.),
        c.get('robust_margin', -1.), c.get('margin', -1.),
        -c.get('command_change_fraction', 0.), c.get('fidelity', 0.)), reverse=True)
    for rank, candidate in enumerate(ordered, 1):
        candidate['functional_rank'] = rank
    return candidates


class LocalSupervisor:
    def __init__(self, endpoint='http://127.0.0.1:8089', model='local-qwen3', timeout=120):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('local supervisor must use a localhost HTTP server')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('unexpected credentials or query in local endpoint')
        self.endpoint, self.model, self.timeout = endpoint.rstrip('/'), model, timeout
        self.audit = []

    def choose(self, observation, candidates):
        ids = [c['id'] for c in candidates if c['ok'] and c['executable'] and c.get('review_eligible', True)]
        if not ids:
            raise ValueError('no allowed agent action')
        schema = dict(type='object', properties=dict(candidate=dict(type='integer', enum=ids),
                      reason=dict(type='string')), required=['candidate', 'reason'], additionalProperties=False)
        system = ('You supervise recovery of a 3D print. Choose one candidate using the supplied tool results. '
            'Only candidates marked ok, executable, and review_eligible are allowed. '
            'The assessment tool ranks them by functional risk: functional_rank 1 is its recommendation. '
            'In policy requests the candidate IDs match their displayed ranks; return that candidate ID. '
            'Aim for a functional completed print. '
            'Prefer continuing when it passes with adequate margin; unnecessary intervention consumes a limited budget. '
            'If fit_at_risk is true, prioritize the largest fit_robust_clearance_mm, then fit_clearance_mm: '
            'a verified recovery with greater clearance '
            'can preserve a through-hole before the next layers damage its fit. '
            'For recovery, consider functional margin, robustness under uncertainty, material change, '
            'and remaining correction distance. margin and robust_margin are dimensionless; '
            'fit_clearance_mm and uncertainty_mm are lengths in millimetres. '
            'Larger margins are better. Negative robust margin means '
            'the uncertainty envelope fails, even if the central prediction passes. '
            'Do not invent measurements or change the functional specification. Reply with JSON and a short reason.')
        keys = ('id', 'name', 'ok', 'executable', 'violations', 'margin', 'robust_margin',
                'fit_clearance_mm', 'fit_robust_clearance_mm', 'fit_at_risk', 'command_change_fraction',
                'uncertainty_mm', 'adapts_remaining', 'fidelity', 'functional_rank', 'review_eligible', 'review_exclusion')
        keys += ('observation_model', 'fallback_after_no_primary_plan')
        keys += ('both_forecasts_pass', 'analytical_violations')

        def compact(value):
            if isinstance(value, float):
                return round(value, 4)
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items() if v is not None and v != []}
            if isinstance(value, list):
                return [compact(v) for v in value]
            return value

        # Full-precision tool results remain in the controller's audit. The
        # small local model receives only decision-relevant fields, with units.
        # Keep every eligible choice. Rejected plans remain in the full audit;
        # omitting them here reduces CPU prompt latency without pruning choices.
        evidence = compact(dict(observation=observation,
            candidates=[{k: c[k] for k in keys if k in c}
                        for c in sorted(candidates, key=lambda c: c.get('functional_rank', 10000))
                        if c['id'] in ids]))
        payload = dict(model=self.model, messages=[dict(role='system', content=system),
            dict(role='user', content=json.dumps(evidence, allow_nan=False, separators=(',', ':')))],
            temperature=0., seed=2026, max_tokens=160,
            chat_template_kwargs=dict(enable_thinking=False),
            response_format=dict(type='json_schema', json_schema=dict(name='recovery_decision', strict=True, schema=schema)))
        record = dict(request=payload, accepted=False, endpoint=self.endpoint, timeout_s=self.timeout)
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
            raise RuntimeError(f'local supervisor failed; refusing to count an unsupervised decision: {type(exc).__name__}: {exc}') from exc
        finally:
            record['seconds'] = time.monotonic() - started
            record['request_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LocalAgentMixin:
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
        considered = rank_for_review(considered, self.state.resolution_mm)
        # The small model confused original index 1 with functional rank 1.
        # Give it one consistent numbering scheme and keep the executable plans
        # in exactly the same order. Its selected ID still determines the action.
        order = sorted(range(len(candidates)), key=lambda i: considered[i].get('functional_rank', 10000+i))
        candidates = [candidates[i] for i in order]
        considered = [considered[i] for i in order]
        for identifier, candidate in enumerate(considered, 1):
            candidate['id'] = identifier
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
            checkpoint_sha256=getattr(getattr(self.state, 'model', None), 'sha256', None))
        selected = self.supervisor.choose(observation, considered)
        chosen = candidates[selected['candidate'] - 1]
        self.last_continue_review = ((len(known), self.supervisor.audit[-1]['request_sha256'])
                                     if chosen['name'] == 'continue' else None)
        considered.append(dict(agent=self.supervisor.audit[-1]))
        # The small model has misstated numerical margins in development. Its
        # prose is preserved in the raw audit, but the operator-facing reason
        # states only the selected action and the tool-verified prediction.
        return Decision('continue' if chosen['name'] == 'continue' else 'adapt',
            'local Qwen selected ' + chosen['name'] + '; predicted function checks pass',
            chosen['plan'], chosen['verdict'], considered)


class AgentNeuralPolicy(LocalAgentMixin, NeuralPolicy):
    """Local supervision with a trained temporal deposition model."""


class AgentContourPolicy(LocalAgentMixin, ContourPolicy):
    """Local supervision with the analytical forecast, e.g. learned-camera ablation."""


class AgentRiskAwareNeuralPolicy(LocalAgentMixin, RiskAwareNeuralPolicy):
    """Local supervision over learned forecasts and preventive fit recovery."""


class AgentFallbackNeuralPolicy(LocalAgentMixin, FallbackNeuralPolicy):
    pass


class AgentFallbackRiskNeuralPolicy(LocalAgentMixin, FallbackRiskNeuralPolicy):
    pass
