import numpy as np
import pytest
from shapely.geometry import box

from tools.boundary_training_augmentation import affected_origins, distance_tile, load_extra_training
from tools.boundary_critical_material import load_critical_material, critical_loss_factors


def example_arrays():
    x = np.zeros((3, 3, 24, 24), dtype='float16')
    x[1, :2] = .25
    y = np.full((3, 1, 64, 64), -.2, dtype='float16')
    split = np.array([False, True, False])
    groups = np.array(['training-a', 'held-out', 'training-b'])
    return x, y, split, groups


def extra_file(tmp_path, **changes):
    x, y, _, _ = example_arrays()
    values = dict(x=x, y=y, group=np.array(['extra-a', 'extra-a', 'extra-b']),
                  parent_group=np.array(['training-a', 'training-a', 'training-b']),
                  validation=np.zeros(3, dtype=bool), parent_data_sha256=np.array('parent'),
                  input_frames=np.array(2), output_upsample=np.array(4), dense_tile_size=np.array(16),
                  input_scales=np.array([1]), include_expected_mask=np.array(True))
    values.update(changes)
    path = tmp_path/'extra.npz'
    np.savez_compressed(path, **values)
    return path


def test_training_append_preserves_held_out_bytes_and_fixed_training_budget(tmp_path):
    original = example_arrays()
    merged = load_extra_training(extra_file(tmp_path), *original, 'parent', np.array([.8, .2]), .5)
    for before, after in zip(original, merged[:4]):
        assert np.array_equal(before, after[:len(before)])
    assert np.flatnonzero(merged[2]).tolist() == [1]
    assert merged[0][merged[2]].tobytes() == original[0][original[2]].tobytes()
    assert merged[1][merged[2]].tobytes() == original[1][original[2]].tobytes()
    # Equal mass per extra trajectory, regardless of its number of crops.
    assert np.allclose(merged[4], [.4, .1, .125, .125, .25])
    assert merged[5]['samples_per_epoch'] == 2
    assert merged[5]['rows'] == 3 and merged[5]['groups'] == 2


@pytest.mark.parametrize('changes,message', [
    ({'parent_data_sha256': np.array('wrong')}, 'different parent'),
    ({'validation': np.array([False, True, False])}, 'training only'),
    ({'parent_group': np.array(['held-out']*3)}, 'parents or group split'),
    ({'parent_group': np.array(['unknown']*3)}, 'parents or group split'),
    ({'group': np.array(['held-out']*3)}, 'parents or group split'),
    ({'input_frames': np.array(1)}, 'incompatible'),
])
def test_augmentation_cannot_import_validation_or_unbound_rows(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_extra_training(extra_file(tmp_path, **changes), *example_arrays(), 'parent', None, .5)


def test_invalid_sampling_and_nonfinite_arrays_are_rejected(tmp_path):
    path = extra_file(tmp_path)
    for fraction in (0, 1, float('nan')):
        with pytest.raises(ValueError, match='fraction'):
            load_extra_training(path, *example_arrays(), 'parent', None, fraction)
    with pytest.raises(ValueError, match='probabilities'):
        load_extra_training(path, *example_arrays(), 'parent', np.array([2., -1.]), .5)
    x, *_ = example_arrays(); x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='arrays'):
        load_extra_training(extra_file(tmp_path, x=x), *example_arrays(), 'parent', None, .5)


def test_native_targets_have_correct_sign_and_clipped_metric_distance():
    target = distance_tile(box(-.5, -.5, .5, .5), (248, 248))
    assert target.dtype == np.float16 and target.shape == (64, 64)
    assert target[32, 32] == .5  # Pixel (256,256) is the bed origin.
    assert target[32, 40] == 0  # Native pixels are 1/16 mm apart.
    assert target[0, 0] == -1
    origins = affected_origins(box(-1, -1, 1, 1), [1, 0], 1)
    assert 1 <= len(origins) <= 4
    assert all(0 <= value <= 496 for origin in origins for value in origin)


def test_combined_labels_bind_both_datasets_and_keep_validation_out_of_weights(tmp_path):
    target = np.array([[[[.1, -.1]]], [[[.1, -.1]]], [[[.1, -.1]]]], dtype='float16')
    groups = np.array(['training', 'validation', 'extra'])
    split = np.array([False, True, False])
    masks = target > 0
    path = tmp_path/'combined.npz'
    np.savez_compressed(path, critical_material=masks.astype('uint8'), group=groups, validation=split,
                        parent_data_sha256=np.array('parent'), extra_data_sha256=np.array('extra-hash'))
    checked = load_critical_material(path, groups, split, target, 'parent', extra_data_sha256='extra-hash')
    factors = critical_loss_factors(target, checked, split, np.array([.25, .75]), .5)
    assert factors['critical_training_pixels'] == 2
    checked[split] = False
    assert factors == critical_loss_factors(target, checked, split, np.array([.25, .75]), .5)
    for wrong in (None, 'different-extra'):
        with pytest.raises(ValueError, match='extra training'):
            load_critical_material(path, groups, split, target, 'parent', extra_data_sha256=wrong)
