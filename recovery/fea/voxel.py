"""Layer stack -> voxel grid. x, y are part-centred mm; z starts at the bed."""
from dataclasses import dataclass
from typing import List

import numpy as np
import shapely

from ..types import Layer


@dataclass
class VoxelGrid:
    solid: np.ndarray            # (nx, ny, nz) bool
    h: float                     # voxel size in x and y (mm)
    hz: float                    # voxel height (mm) = layers_per_voxel * layer height
    x0: float
    y0: float
    z0: float = 0.0


def stack_bounds(layers: List[Layer]):
    b = [l.region.bounds for l in layers if not l.region.is_empty]
    return (min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b))


def voxelize(layers: List[Layer], h=1.0, layers_per_voxel=4, mode="mid", margin=2.0,
             bounds=None) -> VoxelGrid:
    """mode 'mid': a voxel is solid if the middle layer of its group is solid there.
    mode 'min': solid only if EVERY layer of the group is solid (conservative; used for leak
    tests so that a gap in any single layer opens the wall)."""
    m = max(1, int(layers_per_voxel))
    minx, miny, maxx, maxy = bounds or stack_bounds(layers)
    x0 = np.floor((minx - margin) / h) * h
    y0 = np.floor((miny - margin) / h) * h
    nx = int(np.ceil((maxx + margin - x0) / h))
    ny = int(np.ceil((maxy + margin - y0) / h))
    xs = x0 + (np.arange(nx) + 0.5) * h
    ys = y0 + (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xf, yf = X.ravel(), Y.ravel()
    lh = layers[-1].z / len(layers)
    nz = int(np.ceil(len(layers) / m))
    solid = np.zeros((nx, ny, nz), bool)
    cache = {}

    def rast(region):
        key = id(region)
        if key not in cache:
            cache[key] = (np.zeros((nx, ny), bool) if region.is_empty
                          else shapely.contains_xy(region, xf, yf).reshape(nx, ny))
        return cache[key]

    for k in range(nz):
        grp = layers[k * m:(k + 1) * m]
        if mode == "min":
            solid[:, :, k] = np.logical_and.reduce([rast(l.region) for l in grp])
        else:
            solid[:, :, k] = rast(grp[len(grp) // 2].region)
    return VoxelGrid(solid, h, lh * m, float(x0), float(y0), 0.0)
