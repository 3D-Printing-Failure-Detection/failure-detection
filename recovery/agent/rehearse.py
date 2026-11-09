"""Rehearse a real test in software: the same part, a hand dent at a given spot, layer and depth,
on the simulated printer.

A hand dent made while the print is paused is instant and local: the warm top layers are pushed
in at once, and the printer then prints the normal plan on top. So unlike the existing 'collapse'
failure (deform.py), which grows over ~3 mm of height and persists upward, `HandDent` is at full
depth from its first layer and lasts only `layers` layers. It reuses the collapse displacement."""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from ..deform import Failure


@dataclass
class HandDent(Failure):
    layers: int = 5                    # ~1 mm of warm material deforms under the tool

    def active(self, idx: int) -> bool:
        return self.k0 <= idx < self.k0 + self.layers


def hand_dent(nominal, x, y, depth_mm, at_layer, width_mm=3.0, layers=5) -> HandDent:
    """(x, y) in part-centred mm, on or near an outer wall. The push is perpendicular to the
    nearest wall, into the material, at full depth from `at_layer` for `layers` layers."""
    lay = nominal[at_layer]
    edge = lay.region.exterior if lay.region.geom_type == "Polygon" else lay.region.boundary
    c = np.array(nearest_points(edge, Point(x, y))[0].coords[0])
    tip = np.array(nearest_points(lay.region.buffer(-0.5), Point(*c))[0].coords[0])
    n0 = (tip - c) / (np.linalg.norm(tip - c) + 1e-9)
    h = nominal[1].z - nominal[0].z
    # collapse amplitude is min(1, (z - z0 + h) / 3): start the ramp 3 mm below -> full depth at once
    return HandDent("collapse", at_layer, dict(z0=lay.z - 3.0, layer_h=h, c=c.tolist(), n0=n0.tolist(),
                                               sigma=width_mm, severity=depth_mm), layers=layers)


def hand_poke(nominal, x, y, at_layer, hole_mm=2.0, layers=3) -> Failure:
    """Poke through a thin wall at (x, y): material is missing there for `layers` layers (the
    existing under-extrusion failure, thinned past the wall thickness)."""
    lay = nominal[at_layer]
    h = nominal[1].z - nominal[0].z
    return Failure("under_extrusion", at_layer, dict(z0=lay.z - h, layer_h=h, c=[x, y], radius=hole_mm / 2,
                                                    severity=1.0, duration=layers))
