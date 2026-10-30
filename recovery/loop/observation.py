"""Camera-only state and a short-history model of command-to-deposit error.

The uncertainty is a resolution/innovation bound, not a calibrated probability.
No printer internals or injected-failure parameters enter this model.
"""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.adapt import Field
from ..geometry.ops import boundary_samples, clean, warp
from ..types import Layer


class Residual:
    """Translation plus a local boundary displacement, including wall thinning."""
    def __init__(self, command, observed):
        field = Field(command, observed)
        self.translation = field.t
        self.source, self.values = field.src, field.v
        self.tree = cKDTree(self.source) if len(self.source) else None

    def __call__(self, points):
        result = np.broadcast_to(self.translation, points.shape).copy()
        if self.tree is None or not len(points):
            return result
        distance, index = self.tree.query(points + self.translation, k=[1, 2, 3])
        # A short spatial scale keeps opposing inner/outer wall motion distinct.
        valid = index < len(self.values)
        index = np.minimum(index, len(self.values) - 1)
        weights = np.exp(-distance**2 / (2 * .5**2)) * valid
        result += (weights[..., None] * self.values[index]).sum(axis=1) / np.maximum(
            weights.sum(axis=1, keepdims=True), 1e-9)
        return result


class ObservationState:
    def __init__(self, resolution_mm=.25, history=4):
        self.resolution_mm = resolution_mm
        self.residuals = deque(maxlen=history)
        self.observations = []
        self.uncertainty_mm = resolution_mm / 2

    def observe(self, command, measured):
        # Simplify pixel-scale noise without closing gaps or deleting holes. Every
        # layer is retained, including those below the classical alarm threshold.
        estimate = Layer(measured.z, clean(measured.region.simplify(self.resolution_mm / 4)))
        self.observations.append(estimate)
        if command.region.is_empty or estimate.region.is_empty:
            return estimate
        residual = Residual(command, estimate)
        points = boundary_samples(command.region, 2.)
        if self.residuals and len(points):
            innovation = np.linalg.norm(residual(points) - self.residuals[-1](points), axis=1)
            self.uncertainty_mm = max(self.resolution_mm / 2, float(np.percentile(innovation, 90)))
        self.residuals.append(residual)
        estimate.cache["uncertainty_mm"] = self.uncertainty_mm
        return estimate

    def displacement(self, points, ahead=1):
        if not self.residuals:
            return np.zeros_like(points)
        values = np.stack([f(points) for f in self.residuals])
        # Recent median differences reduce camera jitter; extrapolate only three
        # layers, then hold the fault persistent instead of assuming it vanishes.
        trend = np.median(np.diff(values, axis=0), axis=0) if len(values) > 1 else 0.
        trend = np.clip(trend, -.5, .5)
        return values[-1] + min(ahead, 3) * trend

    def predict(self, commands):
        return [Layer(l.z, warp(l.region, lambda p, n=n: self.displacement(p, n)))
                for n, l in enumerate(commands, 1)]

    def compensate(self, desired):
        """Invert estimated deposition error; desired geometry is not a command."""
        out = []
        for n, layer in enumerate(desired, 1):
            def inverse(points, n=n):
                commands = points.copy()
                for _ in range(4):
                    commands = points - self.displacement(commands, n)
                return commands - points
            out.append(Layer(layer.z, warp(layer.region, inverse)))
        return out
