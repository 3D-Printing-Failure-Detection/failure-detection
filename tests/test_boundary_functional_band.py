import numpy as np
import pytest
import json
from pathlib import Path
import shapely
from shapely.geometry import box

from recovery import camera
from recovery.parts import get_part
from tools.boundary_functional_band import functional_boundary_band
from tools.boundary_training_augmentation import distance_tile


def test_functional_band_weights_both_sides_of_physical_thin_wall_and_keeps_old_labels():
    nominal, actual = box(-4, -1, 4, 1), box(-4, -.35, 4, .35)
    origin = np.rint(camera.mm_to_px(np.array([[0., 0.]]))[0]).astype(int)-8
    target = distance_tile(actual, origin)
    original = np.zeros((64, 64), dtype=bool)
    # Retain the earlier supervision alongside the additional physical band.
    original.flat[np.flatnonzero(target > 0)[0]] = True
    selected = functional_boundary_band(actual, nominal, {'min_wall_mm': 1.6}, origin, target, original)
    assert selected.dtype == bool and np.all(selected[original])
    assert np.any(selected & (target > 0)) and np.any(selected & (target < 0))
    assert not np.any(selected & (target < -.2))  # A local band, not all background.
    nominal_target = distance_tile(nominal, origin)
    assert functional_boundary_band(nominal, nominal, {'min_wall_mm': 1.6}, origin,
                                    nominal_target, np.zeros_like(original)).sum() == 0
    np.testing.assert_array_equal(functional_boundary_band(nominal, nominal,
        {'min_wall_mm': 1.6}, origin, nominal_target, original), original)


def test_functional_band_requires_a_spec_and_valid_supervision():
    shape = box(-2, -2, 2, 2)
    target, mask = np.ones((64, 64)), np.zeros((64, 64), dtype=bool)
    for spec in [{}, {'min_wall_mm': float('nan')}, {'min_wall_mm': -1}]:
        with pytest.raises(ValueError, match='physical wall specification'):
            functional_boundary_band(shape, shape, spec, [0, 0], target, mask)
    with pytest.raises(ValueError, match='original critical labels'):
        functional_boundary_band(shape, shape, {'min_wall_mm': 1.6}, [0, 0], -target, ~mask)


def test_valid_deposit_with_self_intersecting_derived_opening_keeps_targets_unchanged():
    fixture = json.loads((Path(__file__).parent/'fixtures/functional_band_overlay.json').read_text())
    actual = shapely.from_wkb(bytes.fromhex(fixture['actual_wkb_hex']))
    part = get_part('mug')
    nominal = part.layers()[fixture['layer']].region
    target = distance_tile(actual, fixture['origin_xy'])
    original_target, original_geometry = target.copy(), actual.wkb
    original = np.zeros((64, 64), dtype=bool)
    original.flat[np.flatnonzero(target > 0)[0]] = True
    assert actual.is_valid
    result = functional_boundary_band(actual, nominal, part.spec, fixture['origin_xy'], target, original)
    assert result.shape == target.shape and result.dtype == bool and np.all(result[original])
    np.testing.assert_array_equal(target, original_target)
    assert actual.wkb == original_geometry
    np.testing.assert_array_equal(result,
        functional_boundary_band(actual, nominal, part.spec, fixture['origin_xy'], target, original))
