"""Adapt the remaining layers to the as-printed shape.

The measured contour M of the current layer k and the planned contour P_k give a
displacement field u (rigid translation + local residual). Remaining planned layers
P_j are warped by u(.) * g(j-k): g decays from 1 (continue exactly on top of what is
there, e.g. circle -> teardrop) to 0 over `decay` layers (blend back to the nominal
design where function allows). Each candidate (decay, gain) is a Plan; the policy
verifies candidates against the function spec before choosing.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.affinity import translate

from ..types import Layer
from .ops import boundary_samples, clean, filled, iou, warp


@dataclass
class Plan:
    name: str
    layers: List[Layer]          # replacement for layers k+1 .. end
    fidelity: float = 0.0        # mean IoU with the nominal remaining layers


class Field:
    """Displacement field fitted between planned contour P and measured contour M."""

    def __init__(self, planned_k, measured_k, sigma=5.0, max_disp=15.0):
        self.sigma = sigma
        fp, fm = filled(planned_k.region), filled(measured_k.region)
        # 1) global rigid translation: minimise XOR area of the filled outlines
        c0 = np.array(fm.centroid.coords[0]) - np.array(fp.centroid.coords[0])

        def cost(t):
            return fp.symmetric_difference(translate(fm, -t[0], -t[1])).area
        r = minimize(cost, c0, method="Nelder-Mead", options=dict(xatol=0.05, fatol=0.5, maxiter=60))
        self.t = r.x if cost(r.x) < cost(np.zeros(2)) else np.zeros(2)
        # 2) local residual from nearest-boundary correspondences
        src = boundary_samples(translate(planned_k.region, *self.t), 1.0)
        dst_pts = boundary_samples(measured_k.region, 0.25)
        if len(src) == 0 or len(dst_pts) == 0:
            self.src, self.v, self.tree = src, np.zeros((0, 2)), None
            return
        _, idx = cKDTree(dst_pts).query(src)
        v = dst_pts[idx] - src
        keep = np.linalg.norm(v, axis=1) < max_disp
        self.src, self.v = src[keep], v[keep]
        self.tree = cKDTree(self.src) if len(self.src) else None

    def local(self, pts):
        if self.tree is None:
            return np.zeros_like(pts)
        lists = self.tree.query_ball_point(pts, 3 * self.sigma)
        out = np.zeros_like(pts)
        for i, nb in enumerate(lists):
            if not nb:
                continue
            d2 = ((self.src[nb] - pts[i]) ** 2).sum(1)
            w = np.exp(-d2 / (2 * self.sigma ** 2))
            out[i] = (w[:, None] * self.v[nb]).sum(0) / (w.sum() + 0.5)
        return out

    def __call__(self, pts, gain=1.0):
        moved = pts + self.t
        return gain * (self.t + self.local(moved))


def make_plan(name, field: Field, remaining_planned: List[Layer], decay: float, gain=1.0,
              relief=None) -> Plan:
    """decay: number of layers over which the correction fades to 0 (inf = never).
    relief: optional fn(nominal_layer) -> geometry to keep open (e.g. a screw hole): it is
    cut out of every adapted layer so a shifted layer cannot narrow the through-hole."""
    out = []
    for n, lay in enumerate(remaining_planned):
        g = 1.0 if not np.isfinite(decay) else max(0.0, 1.0 - n / decay)
        if g <= 1e-3:
            out.append(lay)                       # nominal layer, untouched object
            continue
        reg = warp(lay.region, lambda p, g=g: field(p, gain * g))
        cut = relief(lay) if relief else None
        if cut is not None and not cut.is_empty:
            reg = clean(reg.difference(cut))
        out.append(Layer(lay.z, reg))
    fid = float(np.mean([iou(a.region, b.region) for a, b in zip(out, remaining_planned)])) \
        if out else 1.0
    return Plan(name, out, fid)


def candidate_plans(planned_k: Layer, measured_k: Layer, remaining_planned: List[Layer],
                    decays=(float("inf"), 120, 60, 30, 15), relief=None) -> List[Plan]:
    field = Field(planned_k, measured_k)
    plans = []
    for d in decays:
        tag = "inf" if not np.isfinite(d) else int(d)
        plans.append(make_plan(f"adapt(decay={tag})", field, remaining_planned, d))
        if relief is not None:
            plans.append(make_plan(f"adapt(decay={tag})+hole-relief", field, remaining_planned, d,
                                   relief=relief))
    return sorted(plans, key=lambda p: -p.fidelity)
