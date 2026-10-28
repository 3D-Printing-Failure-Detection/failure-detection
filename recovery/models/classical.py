"""Classical (no-training) segmenter and detector. They define the interface that
the trained networks in models/nets.py implement too, and serve as baselines and
as the default so the whole loop runs without a GPU or any training.

  Segmenter.measure(image, expected_mask) -> shapely region (mm) of the CURRENT layer
  Detector.detect(image, expected_region, measured_region) -> Detection
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .. import camera
from ..geometry.ops import boundary_samples, clean, filled, hole_polys


def hole_area(g):
    return sum(h.area for h in hole_polys(g))


EDGE_BIAS_MM = 0.1
LABELS = ["ok", "collapse", "warp", "shift", "under_extrusion", "detach"]


@dataclass
class Detection:
    label: str
    deviation_mm: float
    info: dict


class ClassicalSegmenter:
    def measure(self, image, expected_mask=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        t1, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = gray > t1
        mask = fg
        if fg.sum() > 100:
            # the current layer is the brightest thing in view; older layers sit at
            # <= ~72% of its brightness. Take the bright core, then grow 1 px (the
            # blurred edge) back into the foreground.
            # c = brightness of the current layer (top tail, so a tiny surviving
            # region still sets it). Older layers sit at <= ~72% of c, so a threshold
            # between the two isolates the current layer; the blur erodes the edges
            # slightly, which EDGE_BIAS_MM puts back.
            g = gray.astype(np.float32)
            c = float(g.max()) / 1.04     # max of the blurred image: robust, and a vanished layer is still found
            mask = g > 0.80 * c
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        region = camera.mask_to_region(mask, simplify=0.0)
        return clean(region.buffer(EDGE_BIAS_MM, quad_segs=4).simplify(0.1)) if EDGE_BIAS_MM else region


def boundary_deviation(a, b, pct=97.0):
    """Robust two-way boundary distance (mm) between two regions."""
    pa, pb = boundary_samples(a, 0.5), boundary_samples(b, 0.5)
    if len(pa) == 0 or len(pb) == 0:
        return 99.0
    d1 = cKDTree(pb).query(pa)[0]
    d2 = cKDTree(pa).query(pb)[0]
    return float(max(np.percentile(d1, pct), np.percentile(d2, pct)))


class ClassicalDetector:
    def __init__(self, dev_thresh_mm=1.0):
        self.th = dev_thresh_mm

    def detect(self, image, expected_region, measured_region) -> Detection:
        e, m = expected_region, measured_region
        Am, Ae = m.area, max(e.area, 1e-6)
        if Am < 0.35 * Ae:
            return Detection("detach", 99.0, dict(area_ratio=Am / Ae))
        dev = boundary_deviation(m, e)
        Ar = Am / Ae
        info = dict(area_ratio=Ar)
        if dev < self.th:
            return Detection("ok", dev, info)
        fe, fm = filled(e), filled(m)
        opn = lambda g: g.buffer(-0.35, quad_segs=4).buffer(0.35, quad_segs=4).area
        a_out, a_in = opn(fm.difference(fe)), opn(fe.difference(fm))
        d_hole = hole_area(m) - hole_area(e)
        dc = float(np.hypot(*(np.array(fm.centroid.coords[0]) - np.array(fe.centroid.coords[0]))))
        info.update(a_out=a_out, a_in=a_in, d_hole=d_hole, centroid_shift=dc)
        if dc > 0.6 * dev and abs(a_out - a_in) < 0.5 * max(a_out, a_in):
            label = "shift"
        elif a_out > 1.5 * a_in:
            label = "warp"
        elif a_in > 1.5 * a_out:
            # a wall that sagged inward shrinks the hole; a starved wall opens it up
            label = "under_extrusion" if d_hole > 0.0 else "collapse"
        else:
            label = "under_extrusion" if Ar < 1 else "warp"
        return Detection(label, dev, info)
