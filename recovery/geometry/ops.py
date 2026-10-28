"""Small shapely helpers used everywhere."""
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def polys(g: BaseGeometry):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for s in g.geoms:
            out.extend(polys(s))
        return out
    return []


def clean(g: BaseGeometry, min_area=0.05) -> BaseGeometry:
    """Make valid, keep polygons only, drop slivers."""
    if g is None or g.is_empty:
        return Polygon()
    if not g.is_valid:
        g = shapely.make_valid(g)
    ps = [p for p in polys(g) if p.area >= min_area]
    if not ps:
        return Polygon()
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


def filled(g: BaseGeometry) -> BaseGeometry:
    """Region with all holes filled."""
    ps = [Polygon(p.exterior) for p in polys(g)]
    if not ps:
        return Polygon()
    return shapely.union_all(ps)


def hole_polys(g: BaseGeometry):
    return [Polygon(r) for p in polys(g) for r in p.interiors]


def warp(g: BaseGeometry, fn, seg=0.6) -> BaseGeometry:
    """Displace every vertex p -> p + fn(p). fn takes/returns an (N,2) array."""
    if g.is_empty:
        return g
    g2 = shapely.segmentize(g, seg)
    g3 = shapely.transform(g2, lambda c: c + fn(c))
    return clean(g3)


def boundary_samples(g: BaseGeometry, step=1.0) -> np.ndarray:
    pts = []
    for p in polys(g):
        for ring in [p.exterior, *p.interiors]:
            n = max(int(ring.length / step), 4)
            d = np.linspace(0, ring.length, n, endpoint=False)
            pts.extend(ring.interpolate(x).coords[0] for x in d)
    return np.array(pts) if pts else np.zeros((0, 2))


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty and b.is_empty:
        return 1.0
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 1.0


def translate(g: BaseGeometry, dx, dy) -> BaseGeometry:
    return shapely.affinity.translate(g, dx, dy)
