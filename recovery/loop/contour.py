"""Experimental boundary-role-aware command-to-deposit prediction.

Match outer boundaries to outer boundaries and holes to holes, instead of
allowing a displacement fit to jump across a thin wall. Missing boundaries are
not reconstructed: an incompatible observation uses the original residual fit,
and the measured prefix remains untouched by the forecast.
"""
import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon

from ..geometry.ops import clean, polys
from ..types import Layer
from .guarded import GuardedPolicy
from .observation import ObservationState, Residual


def _rings(region, interior):
    return [ring for p in polys(region) for ring in (list(p.interiors) if interior else [p.exterior])]


def _samples(ring, step=.5):
    distances = np.linspace(0, ring.length, max(4, int(ring.length / step)), endpoint=False)
    return shapely.get_coordinates(shapely.line_interpolate_point(ring, distances))


def _warp_rings(region, displacement):
    if region.is_empty:
        return region
    parts = []
    for p in polys(shapely.segmentize(region, .6)):
        def moved(ring, interior):
            points = np.asarray(ring.coords)
            return points + displacement(points, interior)
        parts.append(Polygon(moved(p.exterior, False), [moved(r, True) for r in p.interiors]))
    return clean(parts[0] if len(parts) == 1 else MultiPolygon(parts))


class ContourResidual(Residual):
    def __init__(self, command, observed, smoothing_mm=1.):
        super().__init__(command, observed)
        self.smoothing_mm = smoothing_mm
        self.groups = {}
        shifted = translate(command.region, *self.translation)
        for interior in (False, True):
            source_rings, target_rings = _rings(shifted, interior), _rings(observed.region, interior)
            if len(source_rings) != len(target_rings) or not source_rings:
                continue
            # One-to-one association avoids fitting several holes to the same
            # observed hole. Roles and geometry are the only association inputs.
            costs = np.array([[Polygon(a).centroid.distance(Polygon(b).centroid)
                               for b in target_rings] for a in source_rings])
            ii, jj = linear_sum_assignment(costs)
            sources, offsets = [], []
            for i, j in zip(ii, jj):
                source = _samples(source_rings[i])
                target = target_rings[j]
                locations = shapely.line_locate_point(target, shapely.points(source))
                nearest = shapely.get_coordinates(shapely.line_interpolate_point(target, locations))
                sources.append(source)
                offsets.append(nearest - source)
            points, values = np.concatenate(sources), np.concatenate(offsets)
            self.groups[interior] = (cKDTree(points), values)

    def __call__(self, points, interior=None):
        if interior not in self.groups:
            return super().__call__(points)
        tree, values = self.groups[interior]
        result = np.broadcast_to(self.translation, points.shape).copy()
        if not len(points):
            return result
        distances, indices = tree.query(points + self.translation, k=list(range(1, 10)))
        valid = indices < len(values)
        indices = np.minimum(indices, len(values) - 1)
        weights = np.exp(-distances**2 / (2 * self.smoothing_mm**2)) * valid
        result += (weights[..., None] * values[indices]).sum(axis=1) / np.maximum(
            weights.sum(axis=1, keepdims=True), 1e-9)
        return result


class ContourObservationState(ObservationState):
    def __init__(self, *args, shrink_trend=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shrink_trend = shrink_trend
        self.trend_shrinkage_queries = 0

    def observe(self, command, measured):
        estimate = Layer(measured.z, clean(measured.region.simplify(self.resolution_mm / 4)))
        self.observations.append(estimate)
        if command.region.is_empty or estimate.region.is_empty:
            return estimate
        residual = ContourResidual(command, estimate)
        if self.residuals:
            innovations = []
            for interior in (False, True):
                for ring in _rings(command.region, interior):
                    points = _samples(ring, 2.)
                    innovations.extend(np.linalg.norm(residual(points, interior)
                        - self.residuals[-1](points, interior), axis=1))
            if innovations:
                self.uncertainty_mm = max(self.resolution_mm / 2, float(np.percentile(innovations, 90)))
        self.residuals.append(residual)
        estimate.cache['uncertainty_mm'] = self.uncertainty_mm
        return estimate

    def displacement(self, points, ahead=1, interior=None):
        if not self.residuals:
            return np.zeros_like(points)
        values = np.stack([f(points, interior) for f in self.residuals])
        trend = np.median(np.diff(values, axis=0), axis=0) if len(values) > 1 else 0.
        trend = np.clip(trend, -.5, .5)
        if self.shrink_trend and len(values) > 1:
            # A tiny measured increment can create a false topological break
            # when extrapolated. Discount the existing observation uncertainty
            # from the increment's magnitude; retain its direction and the full
            # latest measured deformation. This is not calibrated confidence.
            magnitude = np.linalg.norm(trend, axis=1)
            trend *= np.maximum(0., 1.-self.uncertainty_mm/np.maximum(magnitude, 1e-9))[:, None]
            self.trend_shrinkage_queries += 1
        return values[-1] + min(ahead, 3) * trend

    def predict(self, commands):
        return [Layer(l.z, _warp_rings(l.region, lambda p, interior, n=n:
                                      self.displacement(p, n, interior)))
                for n, l in enumerate(commands, 1)]

    def compensate(self, desired):
        out = []
        for n, layer in enumerate(desired, 1):
            def inverse(points, interior, n=n):
                commands = points.copy()
                for _ in range(4):
                    commands = points - self.displacement(commands, n, interior)
                return commands - points
            out.append(Layer(layer.z, _warp_rings(layer.region, inverse)))
        return out


class ContourPolicy(GuardedPolicy):
    def __init__(self, *args, shrink_trend=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shrink_trend = shrink_trend
        self.state = ContourObservationState(shrink_trend=shrink_trend)
