import numpy as np
import pytest

from tools.add_boundary_command_history import load_previous_commands


def test_previous_commands_require_exact_dataset_and_row_identity(tmp_path):
    path = tmp_path/'previous.npz'
    groups = np.array(['train-case', 'validation-case'])
    split = np.array([False, True])
    masks = np.zeros((2, 1, 24, 24), dtype='uint8')
    masks[1] = 1
    np.savez_compressed(path, previous_expected_masks=masks, group=groups,
                        validation=split, parent_data_sha256=np.array('expected'))
    assert np.array_equal(load_previous_commands(path, groups, split, masks.shape, 'expected'), masks)
    with pytest.raises(ValueError, match='different camera dataset'):
        load_previous_commands(path, groups, split, masks.shape, 'changed')
    with pytest.raises(ValueError, match='row order'):
        load_previous_commands(path, groups[::-1], split, masks.shape, 'expected')
    with pytest.raises(ValueError, match='trajectory split'):
        load_previous_commands(path, groups, ~split, masks.shape, 'expected')
    with pytest.raises(ValueError, match='invalid previous'):
        load_previous_commands(path, groups, split, (2, 1, 16, 16), 'expected')


@pytest.mark.parametrize('value,dtype', [(2, 'uint8'), (1, 'float32')])
def test_previous_commands_reject_nonbinary_or_ambiguous_tiles(tmp_path, value, dtype):
    path = tmp_path/'invalid.npz'
    groups, split = np.array(['case']), np.array([False])
    masks = np.full((1, 1, 24, 24), value, dtype=dtype)
    np.savez(path, previous_expected_masks=masks, group=groups, validation=split,
             parent_data_sha256=np.array('expected'))
    with pytest.raises(ValueError, match='invalid previous'):
        load_previous_commands(path, groups, split, masks.shape, 'expected')
