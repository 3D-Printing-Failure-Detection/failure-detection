"""Retain classical-camera plans, with optional joint checks of corrections.

Both histories contain independent measurements of every actual camera frame.
Neither rewrites an earlier observation. The alternative is assessed against the
same complete functional specification and the same remaining command plan.
The optional joint check compares corrections only when continuation fails.
"""
from ..types import Layer
from ..geometry.spec import Verdict
from .decide import DecisionContext
from .contour import ContourPolicy
from .neural import NeuralPolicy
from .risk_neural import RiskAwareNeuralPolicy


class CameraFallbackMixin:
    def __init__(self, *args, dual_segmenter=None, joint_camera=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual_segmenter = dual_segmenter
        self.joint_camera = joint_camera
        self.joint_camera_checks = 0
        self.alternative = (ContourPolicy(self.spec, self.nominal, self.h,
            allow_adapt=self.allow_adapt, use_fea=self.use_fea,
            max_correction_per_layer_mm=self.max_correction_per_layer_mm)
            if dual_segmenter is not None else None)

    def observe(self, expected, measured, detection, index):
        estimate, detection = super().observe(expected, measured, detection, index)
        if self.alternative is not None:
            if self.dual_segmenter.frames != index+1 or self.dual_segmenter.alternative_region is None:
                raise ValueError('alternative camera measurement is missing or stale')
            alternative = Layer(expected.z, self.dual_segmenter.alternative_region)
            self.alternative.state.observe(expected, alternative)
        return estimate, detection

    def candidates(self, known, current_plan, context=None):
        context = context or DecisionContext()
        primary = [dict(c) for c in super().candidates(known, current_plan, context)]
        for candidate in primary:
            candidate['observation_model'] = 'classical camera'
        primary_passes = any(c['verdict'].ok and c['executable'] for c in primary)
        continue_passes = any(c['name'] == 'continue' and c['verdict'].ok and c['executable'] for c in primary)
        if self.alternative is None or (primary_passes and (not self.joint_camera or continue_passes)):
            return primary
        alternative_known = self.alternative.state.observations
        alternative = [dict(c) for c in self.alternative.candidates(alternative_known, current_plan, context)]
        for candidate in alternative:
            candidate['observation_model'] = 'learned camera'
            candidate['fallback_after_no_primary_plan'] = not primary_passes
        if not primary_passes:
            return primary + alternative
        # A correction that works under only its own measurement model may
        # harm the print. Cross-check both sets of proposals, without merging
        # their histories or changing the original functional specification.
        candidates, joint = [], []
        for source, proposals in ((self, primary), (self.alternative, alternative)):
            other, other_known = ((self.alternative, alternative_known) if source is self else (self, known))
            for original in proposals:
                candidate = dict(original)
                candidate['both_camera_forecasts_pass'] = False
                if candidate['verdict'].ok and candidate['executable']:
                    assessment = other._evaluate(other_known, candidate['plan'], context,
                                                 current_plan, candidate.get('correction_budget'))
                    self.joint_camera_checks += 1
                    candidate['other_camera_violations'] = assessment['verdict'].violations
                    candidate['both_camera_forecasts_pass'] = assessment['verdict'].ok and assessment['executable']
                    if candidate['both_camera_forecasts_pass']:
                        for metric in ('margin', 'robust_margin'):
                            candidate[metric] = min(candidate[metric], assessment[metric])
                        joint.append(candidate)
                candidates.append(candidate)
        if not joint:
            return primary
        for candidate in candidates:
            if candidate['verdict'].ok and not candidate['both_camera_forecasts_pass']:
                candidate['verdict'] = Verdict(False,
                    ['other camera rejects this correction while a jointly passing alternative exists'],
                    candidate['verdict'].metrics)
        return candidates


class FallbackNeuralPolicy(CameraFallbackMixin, NeuralPolicy):
    pass


class FallbackRiskNeuralPolicy(CameraFallbackMixin, RiskAwareNeuralPolicy):
    pass
