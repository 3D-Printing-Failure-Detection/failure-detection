"""Failure injection: how a layer looks when something goes wrong.

Failure kinds
  collapse          a section of wall sags inward (circle -> teardrop); persists upward
  warp              a section curls outward, fading with height
  shift             partial layer shift: everything above z0 is displaced
  under_extrusion   a wall section is thinned for `duration` layers (can open a gap)
  detach            most of the part comes off the bed (unrecoverable -> scrap)
Persisting kinds keep acting on every later layer until the controller adapts.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from shapely.geometry import Point

from .geometry.ops import clean, filled, warp
from .types import Layer

KINDS = ["collapse", "warp", "shift", "under_extrusion", "detach"]


@dataclass
class Failure:
    kind: str
    k0: int                                  # first affected layer index
    p: Dict = field(default_factory=dict)    # kind-specific parameters

    def active(self, idx: int) -> bool:
        if idx < self.k0:
            return False
        if self.kind == "under_extrusion":
            return idx < self.k0 + self.p["duration"]
        return True

    def apply(self, layer: Layer, idx: int) -> Layer:
        p, z0 = self.p, self.p["z0"]
        dz = layer.z - z0
        g = layer.region
        if self.kind in ("collapse", "warp"):
            c, n0, sig, s = np.array(p["c"]), np.array(p["n0"]), p["sigma"], p["severity"]
            h = p["layer_h"]
            if self.kind == "collapse":                  # progressive sag over ~3 mm of height
                amp, sgn = min(1.0, (dz + h) / 3.0), 1.0
            else:                                        # curls up quickly, fades with height
                amp, sgn = min(1.0, (dz + h) / 1.5) * float(np.exp(-max(0.0, dz - 1.5) / 3.0)), -1.0
            def fn(pts):
                w = np.exp(-((pts - c) ** 2).sum(1) / (2 * sig ** 2))
                return (sgn * s * amp * w)[:, None] * n0[None, :]
            g = warp(g, fn)
        elif self.kind == "shift":
            from shapely.affinity import translate
            a = min(1.0, (dz + p["layer_h"]) / (p["ramp_layers"] * p["layer_h"]))
            g = translate(g, a * p["dx"], a * p["dy"])
        elif self.kind == "under_extrusion":
            zone = Point(*p["c"]).buffer(p["radius"])
            thin = g.buffer(-p["severity"], quad_segs=4)       # every wall edge pulled in by e
            g = clean(g.difference(zone).union(thin.intersection(zone)))
        elif self.kind == "detach":
            g = clean(g.intersection(Point(*p["c"]).buffer(p["keep_radius"])))
        return Layer(layer.z, g)


def random_failure(planned: List[Layer], rng: np.random.Generator, kind=None, k0=None,
                   severity=None) -> Failure:
    kind = kind or rng.choice(KINDS[:4])
    n = len(planned)
    # fail from the base-adjacent region up to ~85% of the height
    k0 = int(rng.integers(max(6, int(0.05 * n)), int(0.85 * n))) if k0 is None else k0
    lay = planned[k0]
    fl = filled(lay.region)
    cen = np.array(fl.centroid.coords[0])
    theta = rng.uniform(0, 2 * np.pi)
    ray = np.array([np.cos(theta), np.sin(theta)])
    far = fl.boundary.intersection(
        __import__("shapely").geometry.LineString([cen, cen + ray * 200]))
    pts = [np.array(q.coords[0]) for q in getattr(far, "geoms", [far]) if not q.is_empty]
    c = max(pts, key=lambda q: np.linalg.norm(q - cen)) if pts else cen
    n0 = (cen - c) / (np.linalg.norm(cen - c) + 1e-9)
    lh = planned[1].z - planned[0].z
    p = dict(z0=lay.z - lh, layer_h=lh, c=c.tolist(), n0=n0.tolist(), sigma=float(rng.uniform(5, 10)))
    if kind == "collapse":
        p["severity"] = severity if severity is not None else float(rng.uniform(0.8, 7.0))
    elif kind == "warp":
        p["severity"] = severity if severity is not None else float(rng.uniform(0.8, 4.0))
    elif kind == "shift":
        ang = rng.uniform(0, 2 * np.pi)
        m = severity if severity is not None else float(rng.uniform(0.6, 7.0))
        p.update(dx=float(m * np.cos(ang)), dy=float(m * np.sin(ang)), severity=m,
                 ramp_layers=int(rng.integers(1, 9)))
    elif kind == "under_extrusion":
        p.update(severity=severity if severity is not None else float(rng.uniform(0.3, 1.6)),
                 duration=int(rng.integers(1, 6)), radius=float(rng.uniform(6, 14)))
    elif kind == "detach":
        p.update(keep_radius=float(rng.uniform(1, 4)), severity=30.0)
    return Failure(kind, k0, p)
