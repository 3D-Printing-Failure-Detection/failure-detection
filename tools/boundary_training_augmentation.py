"""Training-only camera examples from varied faults on retained clean deposits.

These are rendered augmentations, not new physical rollouts. Deposited geometry
supplies targets and crop locations; network inputs remain two images and the
issued command. The original validation data is never augmented.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.deform import random_failure
from recovery.models.boundary import normalized_gray
from recovery.parts import get_part
from recovery.types import Layer
from tools.boundary_critical_material import critical_material
from tools.boundary_tiles import tile_inputs, tile_query_pixels
from tools.experiment_provenance import runtime_digest, simulation_contract


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance_tile(region, origin):
    points = camera.px_to_mm(tile_query_pixels(origin, 16, 4).reshape(-1, 2))
    inside = shapely.contains_xy(region, points[:, 0], points[:, 1])
    distances = shapely.distance(region.boundary, shapely.points(points))
    return np.clip(distances*np.where(inside, 1, -1), -1, 1).reshape(64, 64).astype('float16')


def affected_origins(region, center, radius):
    """Four fixed boundary locations in the affected zone, without model scores."""
    points = shapely.get_coordinates(shapely.segmentize(region.boundary, .5))
    points = points[np.linalg.norm(points-np.asarray(center), axis=1) <= radius+2]
    if not len(points):
        raise ValueError('fault zone has no command boundary')
    selected = points[np.linspace(0, len(points)-1, 4).astype(int)]
    origins = np.clip(np.rint(camera.mm_to_px(selected)).astype(int)-8, 0, camera.CAM.size-16)
    return sorted(set(map(tuple, origins.tolist())))


def load_extra_training(path, x, y, split, groups, parent_sha256, sampling, fraction):
    """Append training rows and preserve every original validation byte/index."""
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('extra training fraction must be between zero and one')
    with np.load(path, allow_pickle=False) as data:
        if str(data['parent_data_sha256']) != parent_sha256:
            raise ValueError('extra training data refers to a different parent')
        ex, ey, eg, ep = [data[k] for k in ('x', 'y', 'group', 'parent_group')]
        es = data['validation']
        if es.dtype != bool or es.shape != (len(ex),) or es.any():
            raise ValueError('extra examples must be training only')
        if (int(data['input_frames']) != 2 or int(data['output_upsample']) != 4
                or int(data['dense_tile_size']) != 16 or not bool(data['include_expected_mask'])
                or not np.array_equal(data['input_scales'], [1])):
            raise ValueError('incompatible extra camera inputs')
    if (len(ex) == 0 or ex.shape != (len(ex), 3, 24, 24) or x.shape[1:] != ex.shape[1:]
            or ey.shape != (len(ex), 1, 64, 64) or eg.shape != (len(ex),) or ep.shape != (len(ex),)
            or ex.dtype != x.dtype or ey.dtype != y.dtype or not np.isfinite(ex).all()
            or not np.isfinite(ey).all() or np.max(np.abs(ey)) > 1
            or not np.isin(ex[:, 2], [0, 1]).all()):
        raise ValueError('invalid extra training arrays')
    if set(ep)-set(groups[~split]) or set(ep) & set(groups[split]) or set(eg) & set(groups):
        raise ValueError('extra training parents or group split are invalid')
    n_train = int((~split).sum())
    original = np.full(n_train, 1/n_train) if sampling is None else np.asarray(sampling)
    if (original.shape != (n_train,) or not np.isfinite(original).all()
            or np.any(original < 0) or not np.isclose(original.sum(), 1)):
        raise ValueError('invalid original training probabilities')
    names, inverse, counts = np.unique(eg, return_inverse=True, return_counts=True)
    extra_probability = 1/(len(names)*counts[inverse])
    probability = np.r_[(1-fraction)*original, fraction*extra_probability]
    merged = tuple(np.concatenate([a, b]) for a, b in ((x, ex), (y, ey), (split, es), (groups, eg)))
    assert all(np.array_equal(a[split], b[:len(split)][split]) for a, b in zip((x, y, groups), (merged[0], merged[1], merged[3])))
    metadata = dict(path=str(Path(path).resolve()), sha256=file_hash(path), parent_data_sha256=parent_sha256,
                    rows=len(ex), groups=len(names), parent_groups=sorted(set(ep.tolist())),
                    extra_sampling_mass=fraction, samples_per_epoch=n_train,
                    original_validation_unchanged=True, training_only=True,
                    sampling='half-mass setting is explicit; extra mass is uniform over augmented groups, then tiles')
    return (*merged, probability, metadata)


def build_job(job):
    from recovery.printer.mujoco import MujocoPrinter
    path, expected_hash, name, seed, severities = job
    assert file_hash(path) == expected_hash
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    assert record['validation'] is False and '-clean-' in record['group']
    nominal = get_part(name).layers()
    clean = [Layer(f['z'], shapely.from_wkb(f['actual'])) for f in record['frames']]
    assert len(clean) == len(nominal)
    assert all(f['z'] == layer.z and shapely.from_wkb(f['command']).equals_exact(layer.region, 1e-8)
               for f, layer in zip(record['frames'], nominal))
    xs, ys, rows, groups, summaries = [], [], [], [], []
    for severity in severities:
        failure = random_failure(nominal, np.random.default_rng(seed), kind='under_extrusion', severity=severity)
        identity = f'augmented-{name}-under_extrusion-{seed}-{severity:.2f}'
        actual = [failure.apply(layer, i) if failure.active(i) else layer for i, layer in enumerate(clean)]
        indices = sorted({failure.k0, min(len(actual)-1, failure.k0+failure.p['duration']-1)})
        info = dict(group=identity, parent_group=record['group'], parent_sha256=expected_hash,
                    seed=seed, severity=severity, onset=failure.k0, parameters=failure.p,
                    rendered_frames=0, rows=0, critical_positive_pixels=0, true_open_frames=0)
        for profile in ('cartesian', 'corexy', 'bedslinger'):
            previous_seed = int.from_bytes(hashlib.sha256((identity+profile+'-previous-camera').encode()).digest()[:8], 'little')
            with MujocoPrinter(nominal, None, seed, profile, False) as printer, MujocoPrinter(
                    nominal, None, previous_seed, profile, False) as previous:
                for index in indices:
                    printer.printed = actual[:index+1]
                    image = printer.camera_image()
                    previous.printed = actual[:index]
                    previous_image = previous.camera_image()
                    command = camera.expected_mask(nominal[index]).astype('float32')
                    frames = [normalized_gray(image), normalized_gray(previous_image), command]
                    info['rendered_frames'] += 1
                    polygon_parts = list(shapely.get_parts(actual[index].region))
                    info['true_open_frames'] += int(not any(len(p.interiors) for p in polygon_parts if p.geom_type == 'Polygon'))
                    for origin in affected_origins(nominal[index].region, failure.p['c'], failure.p['radius']):
                        target = distance_tile(actual[index].region, origin)
                        rare = critical_material(actual[index].region, nominal[index].region, origin, target)
                        xs.append(tile_inputs(frames, origin, 16).astype('float16'))
                        ys.append(target[None]); groups.append(identity)
                        rows.append(dict(group=identity, parent_group=record['group'], profile=profile, layer=index,
                                         origin_xy=origin, critical_positive_pixels=int(rare.sum()),
                                         current_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                         previous_image_sha256=hashlib.sha256(previous_image.tobytes()).hexdigest(),
                                         actual_sha256=hashlib.sha256(actual[index].region.wkb).hexdigest()))
                        info['rows'] += 1; info['critical_positive_pixels'] += int(rare.sum())
        summaries.append(info)
    return np.stack(xs), np.stack(ys), groups, rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--severities', type=float, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to('/mnt/data/ishaan'):
        parser.error('choose a fresh output directory on the data volume')
    if (len(set(args.seeds)) != len(args.seeds) or len(set(args.severities)) != len(args.severities)
            or any(not np.isfinite(s) or not .3 <= s <= 1.6 for s in args.severities)):
        parser.error('use distinct training seeds and severities within the original fault range')
    parent = json.loads((args.dataset/'manifest.json').read_text())
    parent_hash = file_hash(args.dataset/'data.npz')
    assert parent['status'] == 'complete' and parent['data_sha256'] == parent_hash
    with np.load(args.dataset/'data.npz', allow_pickle=False) as data:
        split, groups = data['validation'], data['group']
    held_out = {int(group.rsplit('-', 1)[1]) for group in groups[split]}
    assert not set(args.seeds) & (held_out | {1001, 1002, 1003, 1004, 2001, 2002, 2003, 2004})
    hashes = {row['group']: row['trajectory_sha256'] for row in parent['groups']}
    training = set(groups[~split]); jobs = []
    for name in ('cup', 'mug'):
        parents = sorted(g for g in training if f'-{name}-clean-' in g)
        assert parents
        for i, seed in enumerate(args.seeds):
            group = parents[i % len(parents)]
            jobs.append((args.trajectories/(group+'.json.gz'), hashes[group], name, seed, args.severities))
    args.out.mkdir(parents=True)
    manifest = dict(status='running', script_sha256=file_hash(__file__), parent_data_sha256=parent_hash,
                    source_sha256=runtime_digest(Path(__file__).resolve().parents[1]),
                    simulation_contract_sha256=simulation_contract(Path(__file__).resolve().parents[1])['sha256'],
                    expected_groups=len(jobs)*len(args.severities), training_only=True,
                    arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    purpose='Broaden training coverage with surviving thin material and true gaps at varied locations',
                    limitation='Fault-model augmentation of retained clean deposits, not new physical rollouts or successful recoveries')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(build_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(dict(groups=len(result[4]), rows=len(result[0]), first_group=result[2][0])), flush=True)
    results.sort(key=lambda row: row[2][0])
    x = np.concatenate([r[0] for r in results]); y = np.concatenate([r[1] for r in results])
    rows = [row for r in results for row in r[3]]
    group = np.asarray([g for r in results for g in r[2]])
    parent_group = np.asarray([row['parent_group'] for row in rows])
    assert set(parent_group) <= training and not set(parent_group) & set(groups[split])
    np.savez_compressed(args.out/'data.npz', x=x, y=y, group=group, parent_group=parent_group,
                        validation=np.zeros(len(x), dtype=bool), parent_data_sha256=np.array(parent_hash),
                        input_frames=np.array(2), output_upsample=np.array(4), dense_tile_size=np.array(16),
                        input_scales=np.array([1]), include_expected_mask=np.array(True))
    assert file_hash(args.dataset/'data.npz') == parent_hash
    summaries = [g for r in results for g in r[4]]
    assert len(summaries) == manifest['expected_groups']
    manifest.update(status='complete', rows=len(x), data_sha256=file_hash(args.out/'data.npz'),
                    critical_positive_pixels=sum(row['critical_positive_pixels'] for row in rows),
                    groups=summaries, row_records=rows, original_dataset_unchanged=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('groups', 'row_records')}), flush=True)


if __name__ == '__main__':
    main()
