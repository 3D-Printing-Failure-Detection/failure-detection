"""Append known commanded-mask tiles to a verified retained camera dataset.

Replays the dense builder's sample locations and checks every original target.
Observed images, targets and trajectory splits are copied without alteration.
Deposited truth is used only to verify the old sampling locations and labels;
the new input comes exclusively from the captured command for that layer.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.parts import get_part
from recovery.types import Layer
from tools.boundary_tiles import tile_inputs, tile_query_pixels
from tools.experiment_provenance import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sampling-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to('/mnt/data/ishaan'):
        parser.error('choose a fresh output directory on the data volume')
    parent = json.loads((args.dataset/'manifest.json').read_text())
    settings = parent['arguments']
    required = dict(frames=12, pixels=1, all_profiles=True, dense_start=True,
                    output_upsample=4, include_base_frames=True, focus_deformed=True,
                    input_scales=[1], input_frames=2, dense_tile_size=16)
    if parent['status'] != 'complete' or any(settings[key] != value for key, value in required.items()):
        raise ValueError('unsupported source sampling protocol')
    trajectories = Path(settings['trajectories'])
    data_hash = file_digest(args.dataset/'data.npz')
    audit = json.loads(args.sampling_audit.read_text())
    with np.load(args.dataset/'data.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    x, y, groups, split = [arrays[key] for key in ('x', 'y', 'group', 'validation')]
    if (audit['status'] != 'complete' or audit['data_sha256'] != data_hash
            or audit['verified_target_rows'] != len(x) or audit['total_rows'] != len(x)):
        raise ValueError('sampling audit is not bound to the source dataset')
    if x.shape != (len(x), 2, 24, 24) or y.shape != (len(x), 1, 64, 64):
        raise ValueError('unexpected dense tile shape')
    masks = np.empty((len(x), 1, 24, 24), dtype=x.dtype)
    original_groups = {row['group']: row for row in parent['groups']}
    audited_rows = {row['dataset_row']: row for row in audit['rows']}
    replay_rows, records = [], []
    verified = 0
    for group in sorted(set(groups)):
        path = trajectories/(group+'.json.gz')
        digest = file_digest(path)
        assert digest == original_groups[group]['sha256'], 'source trajectory changed'
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        _, name, _, _ = group.split('-')
        part = get_part(name)
        nominal = part.layers()
        actual = [Layer(frame['z'], shapely.from_wkb(frame['actual'])) for frame in record['frames']]
        commands = [Layer(frame['z'], shapely.from_wkb(frame['command'])) for frame in record['frames']]
        assert len(actual) == len(commands) == len(nominal)
        assert all(command.z == layer.z and command.region.equals_exact(layer.region, 1e-8)
                   for command, layer in zip(commands, nominal)), 'retained commands differ from this part'
        eligible = record['eligible']
        count = min(12, len(eligible))
        dense_count = min(10, count)
        positions = list(range(dense_count))
        if count > dense_count:
            positions += np.linspace(dense_count, len(eligible)-1, count-dense_count).astype(int).tolist()
        indices = sorted(set(np.asarray(eligible)[positions].tolist()) |
                         {i for i, layer in enumerate(nominal) if layer.z <= part.spec['container']['base_h']})
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], 'little'))
        saved = np.flatnonzero(groups == group)
        assert np.all(split[saved] == record['validation']), 'trajectory split changed'
        row_index = 0
        for profile in ('cartesian', 'corexy', 'bedslinger'):
            for index in indices:
                layer = actual[index]
                mask = camera.expected_mask(layer)
                border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((9, 9), 'uint8'))
                yy, xx = np.nonzero(border)
                if not len(yy):
                    continue
                difference = (mask != camera.expected_mask(nominal[index])).astype('uint8')
                neighborhood = cv2.dilate(difference, np.ones((9, 9), 'uint8'))
                focus = np.flatnonzero(neighborhood[yy, xx])
                use_focus = bool(len(focus)) and rng.random() < .5
                chosen = rng.choice(focus if use_focus else np.arange(len(yy)))
                origin = (int(np.clip(xx[chosen]-8, 0, camera.CAM.size-16)),
                          int(np.clip(yy[chosen]-8, 0, camera.CAM.size-16)))
                points = camera.px_to_mm(tile_query_pixels(origin, 16, 4).reshape(-1, 2))
                inside = shapely.contains_xy(layer.region, points[:, 0], points[:, 1])
                distance = shapely.distance(layer.region.boundary, shapely.points(points))
                expected = np.clip(distance*np.where(inside, 1, -1), -1, 1).astype('float32').reshape(1, 64, 64).astype('float16')
                i = int(saved[row_index])
                row_index += 1
                assert np.array_equal(expected, y[i]), (group, profile, index, 'target replay differs')
                if i in audited_rows:
                    old = audited_rows[i]
                    assert (old['group'], old['profile'], old['layer'], old['origin_xy'], old['validation']) == (
                        group, profile, index, list(origin), bool(split[i])), 'sampling audit location differs'
                command_mask = (camera.expected_mask(commands[index]) != 0).astype('float32')
                masks[i] = tile_inputs([command_mask], origin, 16)
                replay_rows.append(dict(dataset_row=i, group=group, profile=profile,
                                        layer=int(index), origin_xy=origin, validation=bool(split[i])))
                verified += 1
        assert row_index == len(saved), (group, row_index, len(saved))
        records.append(dict(group=group, trajectory_sha256=digest, rows=row_index))
        print(group, 'verified', row_index, flush=True)
    assert verified == len(x)
    arrays['x'] = np.concatenate([x, masks], axis=1)
    arrays['include_expected_mask'] = np.array(True)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'data.npz', **arrays)
    with np.load(args.out/'data.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['x'][:, :2], x)
        assert np.array_equal(saved['x'][:, 2:], masks)
        for key in arrays.keys()-{'x', 'include_expected_mask'}:
            assert np.array_equal(saved[key], arrays[key]), key
    manifest = dict(status='complete', script_sha256=file_digest(__file__),
        data_sha256=file_digest(args.out/'data.npz'), parent_data_sha256=data_hash,
        parent_manifest_sha256=file_digest(args.dataset/'manifest.json'),
        input='unchanged current and prior observed images, followed by captured commanded layer binary mask',
        actual_geometry_role='verify existing targets and sampling locations only; never an input channel',
        original_images_targets_groups_and_split_bitwise_equal=True,
        commands_verified_against_retained_capture=True, verified_target_rows=verified,
        total_rows=len(x), groups=records, rows=replay_rows)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    audit.update(data_sha256=manifest['data_sha256'], parent_data_sha256=data_hash,
                 parent_sampling_audit_sha256=file_digest(args.sampling_audit),
                 unchanged_data_proof_sha256=file_digest(args.out/'manifest.json'),
                 derivation='All original image channels, targets, groups and splits equal; every target and audited sampling location replayed exactly. Appended command feature does not alter critical training labels.')
    (args.out/'sampling-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Command-input dataset complete:', verified, 'verified tiles', flush=True)


if __name__ == '__main__':
    main()
