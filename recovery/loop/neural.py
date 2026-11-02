"""Learned camera-only temporal forecast inside the existing guarded controller."""
import numpy as np

from ..models.temporal import TemporalModel, features
from ..types import Layer
from .contour import ContourObservationState, ContourPolicy, _rings, _warp_rings


class NeuralObservationState(ContourObservationState):
    def __init__(self, checkpoint, calibrate_measurements=False, **kwargs):
        super().__init__(**kwargs)
        self.model = TemporalModel(checkpoint)
        self.neural_queries = 0
        self.calibrate_measurements = calibrate_measurements
        if calibrate_measurements and not self.model.supports_calibration:
            raise ValueError('measurement calibration requires a checkpoint trained on current-state targets')

    def observe(self, command, measured):
        estimate = super().observe(command, measured)
        if not self.calibrate_measurements or len(self.residuals) < 4:
            return estimate
        topology = lambda region: tuple(len(_rings(region, role)) for role in (False, True))
        if topology(command.region) != topology(estimate.region) or estimate.region.is_empty:
            return estimate
        current = self.residuals[-1]

        def correction(points, interior):
            # Locate the corresponding command coordinates, then correct only
            # the observed boundary by at most one camera pixel. No missing
            # hole or component is reconstructed from the nominal command.
            source = points - current.translation
            for _ in range(3):
                source = points - current(source, interior)
            delta = self.displacement(source, 0, interior) - current(source, interior)
            return delta * np.minimum(1., self.resolution_mm / np.maximum(np.linalg.norm(delta, axis=1), 1e-9))[:, None]

        region = _warp_rings(estimate.region, correction)
        if region.is_empty or topology(region) != topology(estimate.region):
            return estimate
        calibrated = Layer(estimate.z, region)
        calibrated.cache['uncertainty_mm'] = self.uncertainty_mm
        # Replace only this frame's estimate. All earlier measurements and raw
        # camera residual history remain untouched.
        self.observations[-1] = calibrated
        return calibrated

    def displacement(self, points, ahead=1, interior=None):
        if len(self.residuals) < 4 or not len(points):
            return super().displacement(points, ahead, interior)
        values = np.stack([f(points, interior) for f in self.residuals])
        translations = np.stack([f.translation for f in self.residuals])
        self.neural_queries += 1
        return self.model(features(values, translations, ahead, bool(interior)))


class NeuralPolicy(ContourPolicy):
    def __init__(self, *args, checkpoint, calibrate_measurements=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = NeuralObservationState(checkpoint, calibrate_measurements=calibrate_measurements)
