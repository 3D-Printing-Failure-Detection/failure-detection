"""Rehearse a real test in software: the same part, a hand dent at a given spot and depth, on the
simulated printer.

Two ways to dent:
- `hand_dent` (during printing): from a layer up, the wall is pushed in for a few layers.
- `dent_layers` / `poke_layers` (during a pause, as in the real test): the warm top layers that
  are already printed are bent in. The bend is full depth at the top and tapers to nothing about
  1 mm down, because warm plastic bends rather than shearing one layer sideways.
`PlannedPauseSim` stops at a layer like a pause added in Bambu Studio."""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from ..deform import Failure
from ..geometry.ops import clean
from ..sim import SimPrinter
from ..types import Layer


@dataclass
class HandDent(Failure):
    layers: int = 5                    # ~1 mm of warm material deforms under the tool

    def active(self, idx: int) -> bool:
        return self.k0 <= idx < self.k0 + self.layers


def _push(layer, x, y):
    """Point on the outer wall nearest (x, y) and the inward direction there."""
    edge = layer.region.exterior if layer.region.geom_type == "Polygon" else layer.region.boundary
    c = np.array(nearest_points(edge, Point(x, y))[0].coords[0])
    tip = np.array(nearest_points(layer.region.buffer(-0.5), Point(*c))[0].coords[0])
    return c, (tip - c) / (np.linalg.norm(tip - c) + 1e-9)


def hand_dent(nominal, x, y, depth_mm, at_layer, width_mm=3.0, layers=5) -> HandDent:
    """(x, y) in part-centred mm, on or near an outer wall. The push is perpendicular to the
    nearest wall, into the material, at full depth from `at_layer` for `layers` layers."""
    lay = nominal[at_layer]
    c, n0 = _push(lay, x, y)
    h = nominal[1].z - nominal[0].z
    # collapse amplitude is min(1, (z - z0 + h) / 3): start the ramp 3 mm below -> full depth at once
    return HandDent("collapse", at_layer, dict(z0=lay.z - 3.0, layer_h=h, c=c.tolist(), n0=n0.tolist(),
                                               sigma=width_mm, severity=depth_mm), layers=layers)


def dent_layers(top, x, y, depth_mm, width_mm=3.0):
    """Bend already-printed layers `top` (bottom to top) in at (x, y): full depth on the last,
    tapering linearly to zero below. `width_mm` is the bell's sigma along the wall: 3 mm means the
    dent fades out about 6 mm to each side, a 12 mm stretch of wall, which is what real hand dents on
    the tray measured (Omar, 2026-11-18: three presses, up to 12 mm wide)."""
    c, n0 = _push(top[-1], x, y)
    out = []
    for i, lay in enumerate(top):
        amp = (i + 1) / len(top)
        f = Failure("collapse", 0, dict(z0=lay.z - 3.0, layer_h=0.2, c=c.tolist(), n0=n0.tolist(),
                                        sigma=width_mm, severity=depth_mm * amp))
        out.append(f.apply(lay, 0))
    return out


def poke_layers(top, x, y, hole_mm=2.0, reach_mm=3.0):
    """Poke through the wall nearest (x, y): a tool-sized channel from outside, through the wall,
    `reach_mm` deep, missing in the layers `top`."""
    c, n0 = _push(top[-1], x, y)
    hole = LineString([c - n0 * 1.0, c + n0 * reach_mm]).buffer(hole_mm / 2)
    return [Layer(l.z, clean(l.region.difference(hole))) for l in top]


def hand_poke(nominal, x, y, at_layer, hole_mm=2.0, layers=3) -> Failure:
    """Poke through a thin wall at (x, y) during printing (existing under-extrusion failure)."""
    lay = nominal[at_layer]
    h = nominal[1].z - nominal[0].z
    return Failure("under_extrusion", at_layer, dict(z0=lay.z - h, layer_h=h, c=[x, y], radius=hole_mm / 2,
                                                    severity=1.0, duration=layers))


def change_from(description, top):
    """'dent X Y DEPTH' | 'poke X Y' | 'none' -> new shapes for the layers `top`, or None."""
    words = description.split()
    if not words or words[0] == "none":
        return None
    if words[0] == "dent" and len(words) == 4:
        x, y, d = map(float, words[1:])
        return dent_layers(top, x, y, d)
    if words[0] == "poke" and len(words) == 3:
        x, y = map(float, words[1:])
        return poke_layers(top[-3:], x, y)
    raise ValueError(f"cannot read {description!r}: use 'dent X Y DEPTH', 'poke X Y' or 'none' (mm)")


class PlannedPauseSim(SimPrinter):
    """The simulated P1S the agent rehearses on. It stops before layer `pause_at`, like a pause added in
    the slicer (None: no pause); `user_change(description)` is what a person does to the print then.
    Its camera is the printer's own (recovery/p1s_view.py), with the part at bed position `origin`
    (default: plate centre). Like the real printer, it has no top-down view."""

    def __init__(self, *args, pause_at=None, origin=None, p1s_camera=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_at, self._paused_once, self.user_paused = pause_at, False, False
        self.p1s_camera = p1s_camera
        from .. import config as C
        self.origin = tuple(origin or C.BED_CENTER)

    def print_layer(self):
        if self.pause_at is not None and not self._paused_once and self.n_done == self.pause_at:
            self._paused_once = self.user_paused = self.paused = True
            return "paused"
        return super().print_layer()

    def user_change(self, description, taper_layers=5):
        top = self.printed[-taper_layers:]
        new = change_from(description, top)
        if new:
            self.printed[len(self.printed) - len(new):] = new
        return new

    def camera_image(self):
        raise NotImplementedError("the P1S has no top-down view; use snapshot()")

    def snapshot(self):
        """BGR frame from the printer camera: same camera, light and placement every time, the bed at
        the current layer, fresh sensor noise each time."""
        from .. import p1s_view as V
        cam = self.p1s_camera or V.P1SCamera.default()
        n = len(self.printed)
        with V.P1SScene(cam, V.Look(), {"part": V.stack_mesh(self.printed)}, self.origin, 0.0,
                        self.printed[-1].z) as scene:
            rgb, _ = scene.render(["part"])
        self._shots = getattr(self, "_shots", 0) + 1
        rng = np.random.default_rng([n, self._shots])
        return V.Photometric().apply(rgb, rng)[..., ::-1].copy()

    def resume(self):
        self.user_paused = False
        super().resume()

    def replace_remaining(self, layers):
        self.user_paused = False
        super().replace_remaining(layers)
