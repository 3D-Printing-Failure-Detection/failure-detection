import numpy as np
import pytest
import shapely
from shapely.geometry import box

from recovery import camera
from tools.boundary_tiles import tile_query_pixels
from tools.boundary_critical_material import critical_material, load_critical_material, critical_loss_factors


def test_critical_labels_find_only_true_new_thin_material():
    nominal, actual = box(-2, -2, 2, 2), box(-.06, -2, .06, 2)
    origin = (248, 248)
    points = camera.px_to_mm(tile_query_pixels(origin, 16, 4).reshape(-1, 2))
    inside = shapely.contains_xy(actual, points[:, 0], points[:, 1]).reshape(64, 64)
    target = np.where(inside, .03, -.1)
    labels = critical_material(actual, nominal, origin, target)
    assert labels.any() and np.all(inside[labels])
    assert not critical_material(actual, actual, origin, target).any()
    assert not critical_material(actual, nominal, origin, -np.ones_like(target)).any()


def labels_fixture():
    target = np.array([[[[.1, -.1, .8, -.8]]], [[[.2, -.2, .9, -.9]]],
                       [[[.1, -.1, .8, -.8]]]], dtype='float32')
    masks = np.zeros_like(target, dtype=bool)
    masks[0, 0, 0, 0] = True
    validation = np.array([False, False, True])
    return target, masks, validation


def test_training_weights_balance_rare_pixels_under_the_actual_sampler():
    target, masks, split = labels_fixture()
    probabilities = np.array([.8, .2])
    result = critical_loss_factors(target, masks, split, probabilities, .5)
    base = .25+.75*(np.abs(target[:2]) < .5)
    weighted = base*np.where(masks[:2], result['critical_factor'], result['ordinary_factor'])
    rare = probabilities @ (weighted*masks[:2]).mean(axis=(1, 2, 3))
    ordinary = probabilities @ (weighted*~masks[:2]).mean(axis=(1, 2, 3))
    assert rare == pytest.approx(.5) and ordinary == pytest.approx(.5)
    # Altering held-out labels cannot change training factors.
    target[2] = .01
    masks[2] = True
    assert result == critical_loss_factors(target, masks, split, probabilities, .5)
    assert result['critical_training_pixels'] == 1


@pytest.mark.parametrize('mass', [0, 1, -1, float('nan'), float('inf')])
def test_invalid_training_mass_is_rejected(mass):
    target, masks, split = labels_fixture()
    with pytest.raises(ValueError, match='mass'):
        critical_loss_factors(target, masks, split, mass=mass)


def test_missing_critical_training_labels_and_invalid_sampling_are_rejected():
    target, masks, split = labels_fixture()
    with pytest.raises(ValueError, match='both critical'):
        critical_loss_factors(target, np.zeros_like(masks), split)
    for probability in [np.array([1.]), np.array([-.1, 1.1]), np.array([.2, .2]), np.array([np.nan, 1.])]:
        with pytest.raises(ValueError, match='sampling'):
            critical_loss_factors(target, masks, split, probability)


def test_sidecar_binds_labels_to_dataset_order_split_and_actual_material(tmp_path):
    target, masks, split = labels_fixture()
    groups = np.array(['train-a', 'train-b', 'val'])
    path = tmp_path/'labels.npz'
    def save(**changes):
        values = dict(critical_material=masks.astype('uint8'), group=groups,
                      validation=split, parent_data_sha256=np.array('data-hash'))
        values.update(changes)
        np.savez(path, **values)
    save()
    assert np.array_equal(load_critical_material(path, groups, split, target, 'data-hash'), masks)
    for changes, message in [
        ({'parent_data_sha256': np.array('other')}, 'different dataset'),
        ({'group': groups[::-1]}, 'row order'),
        ({'validation': ~split}, 'trajectory split'),
        ({'critical_material': np.ones((3, 2), dtype='uint8')}, 'invalid critical'),
        ({'critical_material': np.full_like(masks, 2, dtype='uint8')}, 'invalid critical'),
        ({'critical_material': np.ones_like(masks, dtype='uint8')}, 'absent material'),
    ]:
        save(**changes)
        with pytest.raises(ValueError, match=message):
            load_critical_material(path, groups, split, target, 'data-hash')
