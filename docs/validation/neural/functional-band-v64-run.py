from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
p = json.loads((root / 'protocol.json').read_text())
sys.path.insert(0, p['source'])
import numpy as np
import shapely
from recovery.deform import Failure
from recovery.parts import get_part
from recovery.types import Layer
from tools.boundary_functional_band import functional_boundary_band
from tools.boundary_training_augmentation import distance_tile
from tools.experiment_provenance import runtime_digest, simulation_contract

read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    assert runtime_digest(p['source']) == p['source_sha256']
    assert simulation_contract(p['source'])['sha256'] == p['simulation_contract_sha256']
    for path, sha in p['pins'].items():
        assert digest(path) == sha, path


def main():
    verify()
    parents = [Path(p['original_data']), Path(p['extra_data'])]
    targets, groups, split = [], [], []
    for path in parents:
        with np.load(path, allow_pickle=False) as data:
            targets.append(data['y']); groups.append(data['group']); split.append(data['validation'])
    original_count = len(targets[0])
    target, groups, split = map(np.concatenate, [targets, groups, split])
    with np.load(p['critical_labels'], allow_pickle=False) as data:
        old = data['critical_material'].astype(bool)
        assert np.array_equal(data['group'], groups) and np.array_equal(data['validation'], split)
        assert str(data['parent_data_sha256']) == digest(parents[0])
        assert str(data['extra_data_sha256']) == digest(parents[1])
    original_manifest, extra_manifest = read(parents[0].parent / 'manifest.json'), read(parents[1].parent / 'manifest.json')
    original_hashes = {row['group']: row['trajectory_sha256'] for row in original_manifest['groups']}
    definitions = {row['group']: row for row in extra_manifest['groups']}
    locations = original_manifest['rows'] + extra_manifest['row_records']
    assert len(locations) == len(target) == 7104
    bands = np.zeros_like(old)
    parts, trajectories, geometries, summaries, records = {}, {}, {}, {}, []
    for i, row in enumerate(locations):
        group, index = row['group'], row['layer']
        assert group == groups[i]
        is_extra = i >= original_count
        if not is_extra:
            assert row['dataset_row'] == i
        parent = row['parent_group'] if is_extra else group
        part_name = parent.split('-')[1]
        if part_name not in parts:
            part = get_part(part_name)
            parts[part_name] = (part, part.layers())
        part, nominal = parts[part_name]
        if parent not in trajectories:
            path = Path(p['trajectories']) / (parent + '.json.gz')
            expected_hash = definitions[group]['parent_sha256'] if is_extra else original_hashes[parent]
            assert digest(path) == expected_hash
            with gzip.open(path, 'rt') as stream:
                trajectories[parent] = json.load(stream)
            assert trajectories[parent]['group'] == parent
        trajectory = trajectories[parent]
        assert bool(trajectory['validation']) == (False if is_extra else bool(split[i]))
        if (group, index) not in geometries:
            frame = trajectory['frames'][index]
            assert frame['z'] == nominal[index].z
            assert shapely.from_wkb(frame['command']).equals_exact(nominal[index].region, 1e-8)
            actual = Layer(frame['z'], shapely.from_wkb(frame['actual']))
            if is_extra:
                definition = definitions[group]
                failure = Failure('under_extrusion', definition['onset'], definition['parameters'])
                assert not split[i] and failure.active(index)
                actual = failure.apply(actual, index)
            geometries[group, index] = actual.region
        actual = geometries[group, index]
        if is_extra:
            assert hashlib.sha256(actual.wkb).hexdigest() == row['actual_sha256']
        reproduced = (np.full((64, 64), -1, dtype='float16') if actual.is_empty
                      else distance_tile(actual, row['origin_xy']))
        assert np.array_equal(reproduced, target[i, 0]), ('target replay mismatch', i)
        bands[i, 0] = functional_boundary_band(actual, nominal[index].region, part.spec,
                                              row['origin_xy'], target[i, 0], old[i, 0])
        assert np.all(bands[i, 0][old[i, 0]])
        count = dict(pixels=int(bands[i].sum()), positive=int((bands[i] & (target[i] > 0)).sum()),
                     negative=int((bands[i] & (target[i] < 0)).sum()), zero=int((bands[i] & (target[i] == 0)).sum()),
                     old_critical=int(old[i].sum()))
        label = 'validation' if split[i] else ('extra_training' if is_extra else 'original_training')
        summary = summaries.setdefault(label, Counter())
        summary.update(count); summary['rows'] += 1; summary['nonzero_rows'] += int(count['pixels'] > 0)
        records.append(dict(dataset_row=i, group=group, layer=index, split=label, **count))
        if (i + 1) % 240 == 0:
            print(json.dumps(dict(verified_rows=i+1, total=len(target))), flush=True)
            (root / 'condition.json').write_text(json.dumps(dict(status='building_labels', verified_rows=i+1, total=len(target)), indent=2)+'\n')
    assert np.all(bands[old]) and not split[original_count:].any()
    np.savez_compressed(root / 'functional-band.npz', functional_band=bands.astype('uint8'), group=groups,
                        validation=split, parent_data_sha256=np.array(digest(parents[0])),
                        extra_data_sha256=np.array(digest(parents[1])), critical_labels_sha256=np.array(digest(p['critical_labels'])))
    with np.load(root / 'functional-band.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['functional_band'].astype(bool), bands)
    (root / 'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    verify()
    result = dict(status='complete', verified_target_rows=len(target), original_rows=original_count,
                  augmented_rows=len(target)-original_count, summaries={k:dict(v) for k,v in summaries.items()},
                  all_original_critical_pixels_preserved=True, original_inputs_targets_and_split_unchanged=True,
                  hard_wall_fraction=.6, wall_spec_mm={name:part.spec['min_wall_mm'] for name,(part,_) in parts.items()},
                  opening_radius_mm={name:.6*part.spec['min_wall_mm']/2 for name,(part,_) in parts.items()},
                  boundary_margin_mm=.15, mask_sha256=digest(root / 'functional-band.npz'), rows_sha256=digest(root / 'rows.jsonl'),
                  source_sha256=p['source_sha256'], protocol_sha256=digest(root / 'protocol.json'),
                  limitation='Supervision audit only. No model is trained, no validation label is used for fitting, and no recovery or accuracy gain is established.')
    (root / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    (root / 'condition.json').write_text(json.dumps(dict(status='complete', manifest_sha256=digest(root / 'manifest.json')), indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (root / 'condition.json').write_text(json.dumps(dict(status='failed', error=repr(exc)), indent=2)+'\n')
        raise
