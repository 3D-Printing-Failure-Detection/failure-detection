"""Memoised per-layer measurements (stored in Layer.cache)."""
from shapely.geometry import Point
from shapely.ops import polylabel

from ..types import Layer
from .ops import hole_polys, polys


def base(layer: Layer) -> dict:
    f = layer.cache.get("base")
    if f is None:
        ps = polys(layer.region)
        main = max(ps, key=lambda p: p.area) if ps else None
        holes = hole_polys(layer.region)
        f = dict(
            area=layer.region.area, n=len(ps), main=main, holes=holes,
            hole_area=sum(h.area for h in holes),
            n_main_holes=len(main.interiors) if main is not None else 0,
        )
        layer.cache["base"] = f
    return f


def thin_frac(layer: Layer, t: float) -> float:
    """Fraction of the layer's material thinner than t (morphological opening)."""
    key = ("thin", round(t, 3))
    v = layer.cache.get(key)
    if v is None:
        a = layer.region.area
        if a <= 0:
            v = 0.0
        else:
            op = layer.region.buffer(-t / 2, quad_segs=4).buffer(t / 2, quad_segs=4)
            v = max(0.0, 1.0 - op.area / a)
        layer.cache[key] = v
    return v


def _unsupported(layer: Layer, prev: Layer, tol=0.6):
    return layer.region.difference(prev.region.buffer(tol, quad_segs=4))


def excess_overhang(layer: Layer, prev: Layer, p_layer: Layer, p_prev: Layer, r=0.5) -> float:
    """Area (mm^2) of material hanging over air that the nominal design does not
    also hang over. Slivers narrower than 2r are ignored."""
    if layer is p_layer and prev is p_prev:
        return 0.0
    key = ("oh", id(prev))
    v = layer.cache.get(key)
    if v is None:
        pk = ("punsup", id(p_prev))
        pu = p_layer.cache.get(pk)
        if pu is None:
            pu = _unsupported(p_layer, p_prev).buffer(0.8, quad_segs=4)
            p_layer.cache[pk] = pu
        ex = _unsupported(layer, prev).difference(pu)
        v = 0.0 if ex.is_empty else ex.buffer(-r, quad_segs=4).buffer(r, quad_segs=4).area
        layer.cache[key] = v
    return v


def inscribed(poly):
    """Centre and diameter of the largest inscribed circle of a polygon."""
    pt = polylabel(poly, tolerance=0.05)
    return (pt.x, pt.y), 2 * poly.boundary.distance(pt)


def hole_near(layer: Layer, centre, radius):
    """Interior polygons of the layer whose centroid is within `radius` of centre."""
    c = Point(*centre)
    return [h for h in base(layer)["holes"] if h.centroid.distance(c) <= radius]
