"""Use learned forecasts to preserve tight fits before a passing plan loses them.

The original guarded policy continues whenever its central forecast passes.
For through-holes, this variant considers recovery when that forecast's clearance
is smaller than the observed residual uncertainty. A still-passing continuation
is replaced only if a feasible candidate improves clearance by more than half a
camera pixel. Existing checks and action limits still apply to every candidate.
"""
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from ..geometry import adapt as A, features as F
from ..geometry.ops import clean, iou
from ..geometry.spec import check
from ..types import Layer
from .decide import DecisionContext, Decision
from .contour import ContourPolicy
from .neural import NeuralPolicy


class PreventiveFitMixin:
    def __init__(self, *args, reserve_fit=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reserve_fit = reserve_fit

    def _targets(self, known):
        targets = list(super()._targets(known))
        yield from targets
        if not self.reserve_fit or not self.spec.get('holes') or not targets:
            return
        hold, budget = targets[0]
        layers = []
        for n, layer in enumerate(hold.layers, 1):
            region = layer.region
            for hole in self.spec['holes']:
                voids = F.hole_near(layer, hole['center'], hole['diameter'][1])
                if not voids:
                    continue
                # Grow the already-present void toward the allowed clearance
                # footprint, at the same explicit per-layer correction rate.
                # This adds reserve in FUTURE layers; it cannot widen the prefix.
                allowed = Point(*hole['center']).buffer(hole['diameter'][1]/2, quad_segs=32)
                opening = unary_union(voids).buffer(n*self.max_correction_per_layer_mm, quad_segs=8).intersection(allowed)
                region = clean(region.difference(opening))
            layers.append(Layer(layer.z, region))
        nominal = self.nominal[len(known):]
        fidelity = float(np.mean([iou(a.region, b.region) for a, b in zip(layers, nominal)]))
        yield A.Plan('bounded hole clearance', layers, fidelity), dict(budget,
            fit_reserve_rate_mm_per_layer=self.max_correction_per_layer_mm)

    def _fit_clearance(self, candidate):
        metrics = candidate['verdict'].metrics
        values = []
        for hole in self.spec.get('holes', []):
            prefix = f"hole_{hole['center']}"
            if prefix + '_d' not in metrics or prefix + '_offset_mm' not in metrics:
                return -1.
            diameter = metrics[prefix + '_d']
            low, high = hole['diameter']
            values.extend([(diameter-low)/2, (high-diameter)/2,
                           hole['tol_center']-metrics[prefix+'_offset_mm']])
        return min(values) if values else -1.

    def _evaluate(self, known, plan, context, current_plan, budget=None):
        candidate = super()._evaluate(known, plan, context, current_plan, budget)
        if self.spec.get('holes'):
            candidate['fit_clearance_mm'] = self._fit_clearance(candidate)
        if candidate['verdict'].ok and self.spec.get('holes'):
            clearances = [self._fit_clearance(candidate)]
            # Retain signed fit slack even when an uncertainty scenario fails.
            # The general margin's -1 sentinel would hide whether a recovery
            # substantially reduces that fit risk. Only the central verdict
            # determines eligibility; these are scenario scores, not guarantees.
            for sign in (-1, 1):
                future = [Layer(l.z, clean(l.region.buffer(sign*self.state.uncertainty_mm, quad_segs=4)))
                          for l in candidate['predicted']]
                verdict = check(known+future, self.nominal, {'holes': self.spec['holes']}, self.h)
                clearances.append(self._fit_clearance({'verdict': verdict}))
            candidate['fit_robust_clearance_mm'] = min(clearances)
        return candidate

    def candidates(self, known, current_plan, context=None):
        context = context or DecisionContext()
        candidates = super().candidates(known, current_plan, context)
        continuation = candidates[0]
        if (not self.spec.get('holes') or not continuation['verdict'].ok or not self.allow_adapt
                or context.adapts_remaining == 0 or len(known) == len(self.nominal)):
            return candidates
        clearance = self._fit_clearance(continuation)
        continuation['fit_clearance_mm'] = clearance
        if clearance >= self.state.uncertainty_mm:
            return candidates
        for target, budget in self._targets(known):
            plan = A.Plan('compensate:' + target.name, self.state.compensate(target.layers), target.fidelity)
            candidate = self._evaluate(known, plan, context, current_plan, budget)
            candidate['fit_clearance_mm'] = self._fit_clearance(candidate)
            candidates.append(candidate)
        improvement = max(c.get('fit_clearance_mm', -1.) for c in candidates
                          if c['verdict'].ok and c['executable']) - clearance
        if improvement <= self.state.resolution_mm / 2:
            return [continuation]
        # An observed-innovation uncertainty, not a calibrated probability.
        continuation['fit_at_risk'] = True
        continuation['clearance_improvement_mm'] = improvement
        return candidates

    def pick_candidates(self, candidates, context):
        if (not self.spec.get('holes') or
                (candidates[0]['verdict'].ok and not candidates[0].get('fit_at_risk'))):
            return super().pick_candidates(candidates, context)
        feasible = [c for c in candidates if c['verdict'].ok and c['executable']]
        if not feasible:
            return super().pick_candidates(candidates, context)
        chosen = max(feasible, key=lambda c: (c.get('fit_robust_clearance_mm', -1.),
                                              c.get('fit_clearance_mm', -1.), c['robust_margin'],
                                              -c['command_change_fraction'], c['plan'].fidelity))
        considered = [{k: v for k, v in c.items() if k not in ('plan', 'predicted', 'verdict', 'stress')}
            | dict(ok=c['verdict'].ok, violations=c['verdict'].violations, fidelity=c['plan'].fidelity,
                   adapts_remaining=context.adapts_remaining) for c in candidates]
        action = 'continue' if chosen['name'] == 'continue' else 'adapt'
        reason = ('passing continuation has the best tested fit robustness' if action == 'continue' else
                  'forecast threatens the fit; a verified recovery adds measurable clearance')
        return Decision(action, reason,
                        chosen['plan'], chosen['verdict'], considered)


class RiskAwareNeuralPolicy(PreventiveFitMixin, NeuralPolicy):
    """Preventive fit planning with the trained temporal forecast."""


class RiskAwareContourPolicy(PreventiveFitMixin, ContourPolicy):
    """Matched planner ablation retaining the analytical contour forecast."""
