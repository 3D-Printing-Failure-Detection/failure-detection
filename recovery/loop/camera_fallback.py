"""Retain a feasible classical-camera plan before consulting a learned view.

Both histories contain independent measurements of every actual camera frame.
Neither rewrites an earlier observation. The alternative is assessed against the
same complete functional specification and the same remaining command plan.
"""
from ..types import Layer
from .contour import ContourPolicy
from .neural import NeuralPolicy
from .risk_neural import RiskAwareNeuralPolicy


class CameraFallbackMixin:
    def __init__(self, *args, dual_segmenter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual_segmenter = dual_segmenter
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
        primary = super().candidates(known, current_plan, context)
        for candidate in primary:
            candidate['observation_model'] = 'classical camera'
        if self.alternative is None or any(c['verdict'].ok and c['executable'] for c in primary):
            return primary
        alternative = self.alternative.candidates(self.alternative.state.observations, current_plan, context)
        for candidate in alternative:
            candidate['observation_model'] = 'learned camera'
            candidate['fallback_after_no_primary_plan'] = True
        return primary + alternative


class FallbackNeuralPolicy(CameraFallbackMixin, NeuralPolicy):
    pass


class FallbackRiskNeuralPolicy(CameraFallbackMixin, RiskAwareNeuralPolicy):
    pass
