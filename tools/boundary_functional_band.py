"""Training-only boundary supervision near the unchanged physical wall limit."""
import numpy as np
import shapely

from recovery import camera
from recovery.geometry.spec import HARD_WALL_FRAC
from tools.boundary_tiles import tile_query_pixels


def functional_boundary_band(region, nominal, spec, origin, target, critical_material, margin_mm=.15):
    """Keep old thin positives and include both sides of a hard-thin boundary.

    Geometry is supervision only. The opening matches geometry.spec's thickness
    scale and polygon resolution. Its difference from the nominal thin material
    focuses weighting on additional thin regions, including nearby absent pixels
    needed to learn their width. The Boolean mask is never a camera input.
    """
    if (target.shape != (64, 64) or critical_material.shape != target.shape
            or critical_material.dtype != bool or np.any(critical_material & ~(target > 0))):
        raise ValueError('invalid functional-band target or original critical labels')
    thickness = spec.get('min_wall_mm')
    if not thickness or not np.isfinite(thickness) or thickness <= 0 or margin_mm != .15:
        raise ValueError('functional band needs a physical wall specification and fixed margin')
    radius = HARD_WALL_FRAC * thickness / 2

    def thin(shape):
        opened = shape.buffer(-radius, quad_segs=4).buffer(radius, quad_segs=4)
        if not opened.is_valid:
            # GEOS can return a self-intersecting opening for valid input near
            # coincident edges. Repair only this derived supervision geometry;
            # require its area (used by the physical rule) to remain unchanged
            # to numerical precision. Actual geometry and targets stay intact.
            repaired = shapely.make_valid(opened)
            if abs(repaired.area-opened.area) > 1e-9*max(1., opened.area):
                raise ValueError('opening repair changes physical thin-area evidence')
            opened = repaired
        return shape.difference(opened)

    band = thin(region).difference(thin(nominal)).buffer(margin_mm, quad_segs=4)
    points = camera.px_to_mm(tile_query_pixels(origin, 16, 4).reshape(-1, 2))
    selected = shapely.contains_xy(band, points[:, 0], points[:, 1]).reshape(64, 64)
    return selected | critical_material
