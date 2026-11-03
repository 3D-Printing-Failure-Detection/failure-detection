"""Consistent camera-patch and continuous-query rotations/reflections."""
import numpy as np


def dihedral_transforms(patch_size, channels, offsets):
    """Return input gathers and transformed XY queries in image coordinates.

    Signed-distance targets keep their row order: transform each query along
    with its patch, rather than pretending off-centre subpixels are invariant.
    """
    if patch_size < 1 or patch_size % 2 != 1 or channels < 1:
        raise ValueError('an odd patch size and positive channel count are required')
    grid = np.arange(patch_size**2).reshape(patch_size, patch_size)
    rotation = np.array([[0, 1], [-1, 0]])  # np.rot90, with image Y pointing down
    permutations, queries = [], []
    for flip in (False, True):
        reflection = np.diag([-1 if flip else 1, 1])
        for turns in range(4):
            permutation = np.rot90(np.fliplr(grid) if flip else grid, turns).ravel()
            permutations.append(np.concatenate([permutation + i*patch_size**2 for i in range(channels)]))
            transform = np.linalg.matrix_power(rotation, turns) @ reflection
            queries.append(np.asarray(offsets) @ transform.T)
    return np.stack(permutations), np.stack(queries).astype('float32')
