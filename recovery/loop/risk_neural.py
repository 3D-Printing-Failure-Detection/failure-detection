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
from ..geometry.ops import clean, hole_polys, iou
from ..geometry.spec import check
from ..types import Layer
from .decide import DecisionContext, Decision
from .contour import ContourPolicy
from .neural import NeuralPolicy


def shared_hole_plan(neural_plan, analytical_plan, nominal):
    """Retain both inverse proposals' voids in future commands only.

    Keep the neural proposal's exterior; subtract the analytical proposal's
    holes without filling, buffering, or inventing any missing opening. This
    proposes commands, not a modified observation. Both full forward checks
    still decide whether the larger voids meet the fit and structural limits.
    """
    if (len(neural_plan.layers) != len(analytical_plan.layers)
            or len(neural_plan.layers) != len(nominal)):
        raise ValueError('hole-envelope proposals must cover the same future layers')
    layers = []
    for neural, analytical, reference in zip(neural_plan.layers, analytical_plan.layers, nominal):
        if neural.z != analytical.z or neural.z != reference.z:
            raise ValueError('hole-envelope proposals must have matching layer heights')
        region = clean(neural.region.difference(unary_union(hole_polys(analytical.region))))
        layers.append(Layer(neural.z, region))
    fidelity = float(np.mean([iou(a.region, b.region) for a, b in zip(layers, nominal)]))
    return A.Plan('shared hole clearance:' + neural_plan.name.removeprefix('compensate:'), layers, fidelity)


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

    def __init__(self, *args, joint_forecast=False, hole_envelope=False, **kwargs):
        super().__init__(*args, **kwargs)
        if hole_envelope and not joint_forecast:
            raise ValueError('hole-envelope recovery requires joint forecasts')
        self.joint_forecast = joint_forecast
        self.hole_envelope = hole_envelope
        self.analytical_candidate_checks = 0

    def candidates(self, known, current_plan, context=None):
        from ..geometry.spec import Verdict
        context = context or DecisionContext()
        primary = super().candidates(known, current_plan, context)
        if (not getattr(self, 'joint_forecast', False) or not hasattr(self.state, 'model') or not self.spec.get('holes')
                or not self.allow_adapt or context.adapts_remaining == 0
                or len(known) == len(self.nominal)):
            return primary
        analytical = RiskAwareContourPolicy(self.spec, self.nominal, self.h,
            allow_adapt=self.allow_adapt, use_fea=self.use_fea,
            max_correction_per_layer_mm=self.max_correction_per_layer_mm,
            reserve_fit=self.reserve_fit)
        analytical.state.residuals = self.state.residuals
        analytical.state.observations = self.state.observations
        analytical.state.uncertainty_mm = self.state.uncertainty_mm
        original_primary = primary
        continuation_assessment = None
        early_warning = len(primary) == 1 and primary[0]['verdict'].ok
        if early_warning:
            # A confident neural continuation can miss the last recoverable
            # layer. Check it before taking the single-candidate fast path.
            self.analytical_candidate_checks += 1
            continuation_assessment = analytical._evaluate(known, primary[0]['plan'],
                context, current_plan, primary[0].get('correction_budget'))
            if (continuation_assessment['verdict'].ok
                    and continuation_assessment['executable']
                    and continuation_assessment['fit_clearance_mm'] >= self.state.uncertainty_mm):
                return original_primary
        targets = list(self._targets(known))
        if early_warning:
            primary = [dict(primary[0], fit_at_risk=True)]
            for target, budget in targets:
                plan = A.Plan('compensate:' + target.name,
                    self.state.compensate(target.layers), target.fidelity)
                primary.append(self._evaluate(known, plan, context, current_plan, budget))
        alternatives = []
        for target, budget in targets:
            plan = A.Plan('analytical inverse:' + target.name,
                analytical.state.compensate(target.layers), target.fidelity)
            alternatives.append(self._evaluate(known, plan, context, current_plan, budget))
        candidates = [dict(c) for c in primary + alternatives]
        joint = []
        def assess_joint(candidate, assessment=None):
            if not candidate['verdict'].ok or not candidate['executable']:
                candidate['both_forecasts_pass'] = False
                return
            if assessment is None:
                self.analytical_candidate_checks += 1
                assessment = analytical._evaluate(known, candidate['plan'], context, current_plan,
                                                  candidate.get('correction_budget'))
            candidate['analytical_violations'] = assessment['verdict'].violations
            candidate['both_forecasts_pass'] = assessment['verdict'].ok and assessment['executable']
            if candidate['both_forecasts_pass']:
                for metric in ('margin', 'robust_margin', 'fit_clearance_mm', 'fit_robust_clearance_mm'):
                    candidate[metric] = min(candidate[metric], assessment[metric])
                if early_warning:
                    # Disagreement alone must not replace a passing neural
                    # continuation with a less robust fit. Compare the joint
                    # candidate bound with the original continuation bound.
                    candidate['early_fit_robustness_preserved'] = (
                        candidate['name'] == 'continue' or
                        candidate['fit_robust_clearance_mm'] >= original_primary[0]['fit_robust_clearance_mm'])
                if candidate.get('early_fit_robustness_preserved', True):
                    joint.append(candidate)

        for index, candidate in enumerate(candidates):
            assess_joint(candidate, continuation_assessment if index == 0 else None)
        if not joint and early_warning and getattr(self, 'hole_envelope', False):
            # Only expand the search after the existing early-recovery choices
            # fail their unchanged joint robustness guard. Previously selected
            # recoveries retain their ranking and behavior.
            neural_plans = {c['name']: c['plan'] for c in primary}
            for (target, budget), alternative in zip(targets, alternatives):
                plan = shared_hole_plan(neural_plans['compensate:' + target.name],
                                        alternative['plan'], self.nominal[len(known):])
                candidate = self._evaluate(known, plan, context, current_plan, budget)
                candidate['hole_envelope'] = True
                candidates.append(candidate)
                assess_joint(candidate)
        if not joint:
            return original_primary
        if early_warning and candidates[0]['both_forecasts_pass']:
            improvement = max(c['fit_clearance_mm'] for c in joint) - candidates[0]['fit_clearance_mm']
            if improvement <= self.state.resolution_mm / 2:
                return original_primary
            candidates[0]['clearance_improvement_mm'] = improvement
        for candidate in candidates:
            if candidate['verdict'].ok and (not candidate['both_forecasts_pass'] or
                    not candidate.get('early_fit_robustness_preserved', True)):
                reason = ('early recovery weakens the passing neural fit robustness' if
                    not candidate.get('early_fit_robustness_preserved', True) else
                    'analytical forecast rejects this plan while a jointly passing alternative exists')
                candidate['verdict'] = Verdict(False,
                    [reason], candidate['verdict'].metrics)
        return candidates


class RiskAwareContourPolicy(PreventiveFitMixin, ContourPolicy):
    """Matched planner ablation retaining the analytical contour forecast."""
