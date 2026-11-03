"""Compare geometric camera errors on retained validation trajectories.

The same deposited geometry is re-rendered through each camera profile. This is
camera validation, not a new physical rollout or a closed-loop success rate.
"""
import argparse
from contextlib import nullcontext
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
from recovery.eval.mujoco_eval import source_digest
from recovery.geometry.ops import clean
from recovery.models.boundary import BoundarySegmenter
from recovery.models.classical import ClassicalSegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.inspect_camera_case import layer_features

FLAGS = ('open_wall', 'base_problem', 'connection_problem', 'soft_thin', 'hard_thin')


def frame_indices(record, part, count):
    eligible = record['eligible']
    count = min(count, len(eligible))
    dense = min(10, count)
    positions = list(range(dense))
    if count > dense:
        positions += np.linspace(dense, len(eligible)-1, count-dense).astype(int).tolist()
    selected = {eligible[i] for i in positions}
    selected.update(i for i, layer in enumerate(part.layers()) if layer.z <= part.spec['container']['base_h'])
    return sorted(selected)


def inspect_group(job):
    path, count, checkpoints = job
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    assert record['validation'], 'training trajectory cannot enter validation'
    _, name, _, seed = record['group'].split('-')
    part = get_part(name)
    nominal = part.layers()
    actual = [Layer(frame['z'], shapely.from_wkb(frame['actual'])) for frame in record['frames']]
    assert len(actual) == len(nominal)
    assert all(frame['z'] == layer.z and shapely.from_wkb(frame['command']).equals_exact(layer.region, 1e-8)
               for frame, layer in zip(record['frames'], nominal)), 'retained commands do not match this source'
    models = {'classical': ClassicalSegmenter()}
    for name, checkpoint, digest in checkpoints:
        models[name] = BoundarySegmenter(checkpoint)
        assert models[name].sha256 == digest, 'checkpoint changed before validation'
    indices = frame_indices(record, part, count)
    rows = []
    needs_history = any(getattr(model, 'input_frames', 1) == 2 for model in models.values())
    for profile in ('cartesian', 'corexy', 'bedslinger'):
        previous_seed = int.from_bytes(hashlib.sha256(
            (record['group'] + profile + '-previous-camera').encode()).digest()[:8], 'little')
        previous_context = (MujocoPrinter(nominal, None, previous_seed, profile, False)
                            if needs_history else nullcontext(None))
        with MujocoPrinter(nominal, None, int(seed), profile, False) as printer, previous_context as previous_printer:
            for index in indices:
                printer.printed = actual[:index+1]
                image = printer.camera_image()
                previous_image = image
                if needs_history and index > 0:
                    previous_printer.printed = actual[:index]
                    previous_image = previous_printer.camera_image()
                expected = camera.expected_mask(nominal[index])
                truth = layer_features(actual[index], nominal[index], part.spec)
                measured = {}
                for name, model in models.items():
                    try:
                        if getattr(model, 'input_frames', 1) == 2:
                            # Selected frames can be far apart. Prime from the
                            # actual immediate predecessor, never the last
                            # selected validation frame or another profile.
                            model.prime_history(previous_image,
                                camera.expected_mask(nominal[max(index-1, 0)]))
                        region = clean(model.measure(image, expected).simplify(.25/4))
                    except Exception as exc:
                        raise RuntimeError(f"{record['group']}/{profile}/layer-{index}/{name}: camera measurement failed") from exc
                    measured[name] = layer_features(Layer(nominal[index].z, region), nominal[index], part.spec)
                rows.append(dict(group=record['group'], profile=profile, layer=index,
                                 truth=truth, measured=measured))
    return rows, dict(group=record['group'], frames=len(rows),
                      previous_camera_images=3*sum(index > 0 for index in indices) if needs_history else 0,
                      trajectory_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections', type=Path, nargs='+', required=True)
    parser.add_argument('--validation-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--expected-groups', type=int, required=True)
    parser.add_argument('--parts', nargs='+', choices=['cup', 'mug'], default=['cup', 'mug'])
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.workers < 1:
        parser.error('frames and workers must be positive')
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')) or args.out.exists():
        parser.error('choose a fresh output directory on the data volume')
    checkpoints, names = [], {'classical'}
    for item in args.checkpoint:
        name, separator, path = item.partition('=')
        if not separator or not name or name in names:
            parser.error('checkpoints need unique LABEL=PATH entries')
        names.add(name)
        checkpoints.append((name, str(Path(path).resolve()), hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    paths, groups, expected_frames = [], set(), 0
    for collection in args.collections:
        parent_manifest = json.loads((collection.parent/'manifest.json').read_text())
        assert parent_manifest['status'] == 'complete'
        for path in sorted(collection.glob('*.json.gz')):
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            _, part_name, _, seed = record['group'].split('-')
            if not record['validation'] or part_name not in args.parts:
                continue
            assert int(seed) in args.validation_seeds, 'unexpected validation seed'
            assert record['group'] not in groups, 'duplicate validation trajectory'
            groups.add(record['group'])
            paths.append(path)
            expected_frames += 3 * len(frame_indices(record, get_part(part_name), args.frames))
    assert len(paths) == args.expected_groups, 'incomplete validation trajectory grid'
    training_manifests = {}
    for name, path, digest in checkpoints:
        training_path = Path(path).parent/'training.json'
        training = json.loads(training_path.read_text())
        assert training['checkpoint_sha256'] == digest, 'training manifest does not identify this model'
        assert not groups & set(training['training_groups']), 'training trajectory entered camera validation'
        training_manifests[name] = dict(path=str(training_path),
            sha256=hashlib.sha256(training_path.read_bytes()).hexdigest())
    args.out.mkdir(parents=True)
    manifest = dict(status='running', source_sha256=source_digest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_script_sha256=hashlib.sha256((Path(__file__).parent/'inspect_camera_case.py').read_bytes()).hexdigest(),
        groups=sorted(groups), expected_frames=expected_frames, checkpoints=checkpoints,
        training_manifests=training_manifests,
        temporal_camera_replay='Two-frame models receive the actual immediate predecessor, with independent camera noise; current frames stay matched across models',
        arguments={k: [str(p) for p in v] if k == 'collections' else str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        limitation='Selected validation camera frames and per-layer flags; not complete-print or recovery success rates')
    manifest_path = args.out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary, frames, records = {}, 0, []
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(inspect_group, (p, args.frames, checkpoints)) for p in paths]
            with (args.out/'frames.jsonl').open('w') as stream:
                for future in as_completed(futures):
                    rows, record = future.result()
                    records.append(record)
                    for row in rows:
                        stream.write(json.dumps(row)+'\n')
                        frames += 1
                        for name, measured in row['measured'].items():
                            flags = summary.setdefault(name, {})
                            for flag in FLAGS:
                                if flag not in row['truth']:
                                    continue
                                counts = flags.setdefault(flag, dict(true_positive=0, true_negative=0,
                                                                    false_positive=0, false_negative=0))
                                truth, predicted = row['truth'][flag], measured[flag]
                                key = ('true_' if truth == predicted else 'false_') + ('positive' if predicted else 'negative')
                                counts[key] += 1
                    stream.flush()
                    print(json.dumps(record), flush=True)
        assert frames == expected_frames
        assert source_digest() == manifest['source_sha256']
        for _, path, digest in checkpoints:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
        manifest.update(status='complete', frames=frames, completed_groups=records)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), completed_groups=records, frames=frames)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
