import numpy as np
import pytest

from recovery.models.boundary import camera_patches
from tools.boundary_tiles import tile_inputs, tile_query_pixels


def test_dense_tiles_match_runtime_patches_and_native_coordinates():
    rng = np.random.default_rng(51)
    frames = [rng.random((24, 32), dtype=np.float32) for _ in range(2)]
    for origin in [(0, 0), (16, 8), (3, 2)]:
        tile = tile_inputs(frames, origin, 16)
        windows = np.lib.stride_tricks.sliding_window_view(tile, (9, 9), axis=(-2, -1))
        dense = windows.transpose(1, 2, 0, 3, 4).reshape(-1, 162)
        grid = tile_query_pixels(origin, 16, 1).reshape(-1, 2)
        reference = np.concatenate([camera_patches(frame, grid) for frame in frames], axis=1)
        np.testing.assert_array_equal(dense, reference)
        fine = tile_query_pixels(origin, 16, 4)
        np.testing.assert_array_equal(fine[::4, ::4], grid.reshape(16, 16, 2))
        np.testing.assert_array_equal(fine[-1, -1], np.asarray(origin)+15.75)
    with pytest.raises(ValueError, match='outside'):
        tile_inputs(frames, (-1, 0), 16)
