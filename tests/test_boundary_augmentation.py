import numpy as np
import pytest

from tools.boundary_augmentation import dihedral_transforms


def test_every_rotated_query_matches_the_rotated_patch_pixel():
    yy, xx = np.mgrid[-4:5, -4:5]
    offsets = np.column_stack([xx.ravel(), yy.ravel()])
    permutations, queries = dihedral_transforms(9, 2, offsets)
    # Each pixel has a unique identity, including the second context scale.
    image = np.arange(162)
    for gather, moved in zip(permutations, queries):
        augmented = image[gather].reshape(2, 9, 9)
        x, y = (moved + 4).astype(int).T
        for channel in range(2):
            assert np.array_equal(augmented[channel, y, x], np.arange(81) + channel*81)


def test_off_centre_queries_rotate_with_their_distance_labels():
    offsets = np.array([[.25, .75], [.5, 0.], [0., 0.]])
    centre = np.array([[1., -2.]])
    _, queries = dihedral_transforms(9, 1, np.concatenate([offsets, centre]))
    expected = .4 - np.linalg.norm(offsets - centre, axis=1)
    for transformed in queries:
        assert np.allclose(.4 - np.linalg.norm(transformed[:-1] - transformed[-1], axis=1), expected)
    assert np.array_equal(queries[1, 0], [.75, -.25])
    assert np.array_equal(queries[4, 0], [-.25, .75])
    with pytest.raises(ValueError):
        dihedral_transforms(8, 1, offsets)
