"""Reproduce thin-material labels for a verified training augmentation.

This extends the original label sidecar; every old label and validation row is
retained, and the new sidecar binds both input datasets by SHA-256.
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
from recovery.deform import Failure
from recovery.parts import get_part
from recovery.types import Layer
from tools.boundary_critical_material import critical_material, load_critical_material
from tools.boundary_training_augmentation import distance_tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-data', type=Path, required=True)
    parser.add_argument('--original-labels', type=Path, required=True)
    parser.add_argument('--augmentation', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to('/mnt/data/ishaan'):
        parser.error('choose a fresh output directory on the data volume')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent_hash, extra_hash = digest(args.original_data), digest(args.augmentation/'data.npz')
    with np.load(args.original_data, allow_pickle=False) as data:
        y, groups, split = [data[k] for k in ('y', 'group', 'validation')]
    original = load_critical_material(args.original_labels, groups, split, y, parent_hash)
    manifest = json.loads((args.augmentation/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['training_only']
    assert manifest['parent_data_sha256'] == parent_hash and manifest['data_sha256'] == extra_hash
    with np.load(args.augmentation/'data.npz', allow_pickle=False) as data:
        ey, eg, es, ep = [data[k] for k in ('y', 'group', 'validation', 'parent_group')]
        assert str(data['parent_data_sha256']) == parent_hash
    assert not es.any() and set(ep) <= set(groups[~split]) and not set(ep) & set(groups[split])
    assert len(ey) == len(manifest['row_records']) == manifest['rows']
    definitions = {g['group']: g for g in manifest['groups']}
    extra_masks = np.zeros_like(ey, dtype=bool)
    retained, nominal, actual, pins = {}, {}, {}, {}
    for i, row in enumerate(manifest['row_records']):
        identity, parent, index = row['group'], row['parent_group'], row['layer']
        assert eg[i] == identity and ep[i] == parent
        definition = definitions[identity]
        assert definition['parent_group'] == parent
        name = parent.split('-')[1]
        if parent not in retained:
            path = args.trajectories/(parent+'.json.gz')
            pins[str(path)] = digest(path)
            assert pins[str(path)] == definition['parent_sha256']
            with gzip.open(path, 'rt') as stream:
                retained[parent] = json.load(stream)
            assert retained[parent]['validation'] is False
        if name not in nominal:
            nominal[name] = get_part(name).layers()
        if (identity, index) not in actual:
            frame = retained[parent]['frames'][index]
            command = nominal[name][index]
            assert frame['z'] == command.z and shapely.from_wkb(frame['command']).equals_exact(command.region, 1e-8)
            failure = Failure('under_extrusion', definition['onset'], definition['parameters'])
            assert failure.active(index)
            actual[identity, index] = failure.apply(Layer(frame['z'], shapely.from_wkb(frame['actual'])), index)
        region = actual[identity, index].region
        assert hashlib.sha256(region.wkb).hexdigest() == row['actual_sha256']
        target = distance_tile(region, row['origin_xy'])
        assert np.array_equal(target, ey[i, 0]), 'augmented target replay differs'
        labels = critical_material(region, nominal[name][index].region, row['origin_xy'], target)
        assert int(labels.sum()) == row['critical_positive_pixels']
        extra_masks[i, 0] = labels
    combined = np.concatenate([original, extra_masks])
    merged_groups, merged_split, merged_y = [np.concatenate([a, b]) for a,b in ((groups, eg), (split, es), (y, ey))]
    assert np.array_equal(combined[:len(original)], original)
    assert int(extra_masks.sum()) == manifest['critical_positive_pixels']
    args.out.mkdir(parents=True)
    output = args.out/'critical-material.npz'
    np.savez_compressed(output, critical_material=combined.astype('uint8'), group=merged_groups,
                        validation=merged_split, parent_data_sha256=np.array(parent_hash), extra_data_sha256=np.array(extra_hash))
    checked = load_critical_material(output, merged_groups, merged_split, merged_y, parent_hash,
                                     extra_data_sha256=extra_hash)
    assert np.array_equal(checked, combined)
    assert digest(args.original_data) == parent_hash and digest(args.augmentation/'data.npz') == extra_hash
    report = dict(status='complete', rows=len(combined), verified_augmented_rows=len(ey),
                  parent_data_sha256=parent_hash, extra_data_sha256=extra_hash,
                  original_labels_sha256=digest(args.original_labels), sidecar_sha256=digest(output),
                  augmentation_manifest_sha256=digest(args.augmentation/'manifest.json'),
                  script_sha256=digest(__file__), trajectory_pins=pins,
                  critical_training_pixels=int(combined[~merged_split].sum()),
                  critical_validation_pixels=int(combined[merged_split].sum()),
                  original_labels_unchanged=True, original_validation_unchanged=True,
                  use='Training loss weights only; no deposited geometry or fault identity enters model inputs')
    (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trajectory_pins'}), flush=True)


if __name__ == '__main__':
    main()
