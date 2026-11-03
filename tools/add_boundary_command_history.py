"""Store prior captured command masks alongside an unchanged camera dataset.

The sidecar avoids duplicating the retained images and labels. It is tied to
their exact compressed file hash and row order. No deposited geometry is read
as a model input; both masks come from commands already issued to the printer.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.types import Layer
from tools.boundary_tiles import tile_inputs


def load_previous_commands(path, groups, split, expected_shape, data_sha256):
    with np.load(path, allow_pickle=False) as data:
        if str(data['parent_data_sha256']) != data_sha256:
            raise ValueError('previous commands refer to a different camera dataset')
        if not np.array_equal(data['group'], groups) or not np.array_equal(data['validation'], split):
            raise ValueError('previous command row order or trajectory split differs')
        masks = data['previous_expected_masks']
    if masks.shape != expected_shape or masks.dtype != np.uint8 or not np.isin(masks, [0, 1]).all():
        raise ValueError('invalid previous command mask tiles')
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to('/mnt/data/ishaan'):
        parser.error('choose a fresh output directory on the data volume')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    data_hash = digest(args.dataset/'data.npz')
    assert manifest['status'] == 'complete' and manifest['data_sha256'] == data_hash
    assert manifest['original_images_targets_groups_and_split_bitwise_equal']
    with np.load(args.dataset/'data.npz', allow_pickle=False) as data:
        x, groups, split = [data[k] for k in ['x', 'group', 'validation']]
        assert int(data['input_frames']) == 2 and bool(data['include_expected_mask'])
    assert x.shape == (len(x), 3, 24, 24)
    assert set(row['dataset_row'] for row in manifest['rows']) == set(range(len(x)))
    assert len(manifest['rows']) == len(x)
    assert not set(groups[split]) & set(groups[~split])
    masks = np.empty((len(x), 1, 24, 24), dtype='uint8')
    pins, cold_starts, changed, seen = {}, 0, 0, set()
    for group in manifest['groups']:
        name = group['group'];path = args.trajectories/(name+'.json.gz')
        pins[str(path)] = digest(path)
        assert pins[str(path)] == group['trajectory_sha256']
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        assert record['group'] == name
        commands = [Layer(frame['z'], shapely.from_wkb(frame['command'])) for frame in record['frames']]
        rows = [row for row in manifest['rows'] if row['group'] == name]
        assert len(rows) == group['rows']
        for row in rows:
            i, index, origin = row['dataset_row'], row['layer'], row['origin_xy']
            assert i not in seen, 'duplicate command-history row'
            seen.add(i)
            assert groups[i] == name and split[i] == row['validation'] == record['validation']
            assert 0 <= index < len(commands)
            assert len(origin) == 2 and all(type(v) is int and 0 <= v <= camera.CAM.size-16 for v in origin)
            current = tile_inputs([camera.expected_mask(commands[index]).astype('float32')], origin, 16)
            assert np.array_equal(current, x[i, 2:3]), 'current command does not reproduce the retained input'
            previous = tile_inputs([camera.expected_mask(commands[max(index-1, 0)]).astype('float32')], origin, 16)
            assert np.isin(previous, [0, 1]).all()
            masks[i] = previous.astype('uint8')
            cold_starts += index == 0
            changed += not np.array_equal(previous, current)
        print(name, len(rows), 'command pairs verified', flush=True)
    assert seen == set(range(len(x))), 'incomplete command-history rows'
    args.out.mkdir(parents=True)
    destination = args.out/'previous-command-masks.npz'
    np.savez_compressed(destination, previous_expected_masks=masks, group=groups,
                        validation=split, parent_data_sha256=np.array(data_hash))
    assert np.array_equal(load_previous_commands(destination, groups, split, masks.shape, data_hash), masks)
    assert digest(args.dataset/'data.npz') == data_hash
    result = dict(status='complete', script_sha256=digest(__file__),
        parent_data=str((args.dataset/'data.npz').resolve()), parent_data_sha256=data_hash,
        parent_manifest_sha256=digest(args.dataset/'manifest.json'),
        sidecar_sha256=digest(destination), rows=len(x), cold_start_rows=cold_starts,
        changed_command_tiles=changed, current_command_tiles_reproduced_exactly=len(x),
        trajectory_pins=pins, training_groups=sorted(set(groups[~split].tolist())),
        validation_groups=sorted(set(groups[split].tolist())),
        input='Actual previous captured command mask, with the current command duplicated at layer zero',
        original_dataset_unchanged=True, deposited_geometry_used_for_inputs=False,
        limitation='A verified extra known input; no training, camera-accuracy or recovery improvement is implied.')
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'trajectory_pins'}), flush=True)


if __name__ == '__main__':
    main()
