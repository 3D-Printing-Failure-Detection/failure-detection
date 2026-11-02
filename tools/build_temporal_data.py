"""Camera-observed synthetic trajectories for temporal displacement training.

The target uses simulated geometry; features use only past rendered measurements.
Whole seeds are assigned to a split before generation. No images are retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.deform import random_failure
from recovery.eval.mujoco_eval import source_digest
from recovery.loop.contour import ContourResidual, _rings, _samples
from recovery.models.classical import ClassicalSegmenter
from recovery.models.temporal import features
from recovery.parts import get_part
from recovery.types import Layer


def trajectory(job):
    name, kind, seed, validation, snapshots = job[:5]
    domain, profile = job[5:7] if len(job) > 5 else ('synthetic', 'cartesian')
    include_current = job[7] if len(job) > 7 else False
    capture_dir = Path(job[8]) if len(job) > 8 and job[8] else None
    started = time.time()
    nominal = get_part(name).layers()
    failure = None if kind == 'clean' else random_failure(nominal, np.random.default_rng(seed), kind=kind)
    identity = f'{profile}-{name}-{kind}-{seed}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little'))
    segmenter = ClassicalSegmenter()
    actual, measured, targets, measured_layers = {}, {}, {}, {}

    def geometry(i):
        if i not in actual:
            actual[i] = failure.apply(nominal[i], i) if failure and failure.active(i) else nominal[i]
        return actual[i]

    def observation(i):
        if i not in measured:
            image = camera.render([geometry(j) for j in range(max(0, i-2), i+1)], rng)
            region = segmenter.measure(image)
            # Match the runtime state's pixel-noise simplification exactly.
            from recovery.geometry.ops import clean
            layer = Layer(nominal[i].z, clean(region.simplify(.25 / 4)))
            measured_layers[i] = layer
            measured[i] = ContourResidual(nominal[i], layer)
        return measured[i]

    onset = failure.k0 if failure else 4
    eligible = np.arange(max(3, onset - 3), len(nominal) - 1)
    if len(eligible) > snapshots:
        # Capture onset densely and sample the remainder without using labels.
        dense = eligible[:min(10, snapshots)]
        rest = rng.choice(eligible[len(dense):], snapshots-len(dense), replace=False)
        eligible = np.sort(np.concatenate([dense, rest]))
    if domain == 'mujoco':
        from recovery.printer.mujoco import MujocoPrinter
        from recovery.geometry.ops import clean
        required = {j for i in eligible for j in range(i-3, i+1)}
        # Training labels may use truth; the saved feature rows still contain
        # only the same camera-derived residuals available at runtime.
        with MujocoPrinter(nominal, failure, seed, profile, False) as printer:
            for i in range(len(nominal)):
                actual[i] = printer.print_layer()
                image = printer.camera_image()
                if i in required:
                    region = segmenter.measure(image)
                    measured_layers[i] = Layer(nominal[i].z, clean(region.simplify(.25 / 4)))
                    measured[i] = ContourResidual(nominal[i], measured_layers[i])
    xs, ys = [], []
    for i in eligible:
        history = [observation(j) for j in range(i-3, i+1)]
        for ahead in ((0,) if include_current else ()) + (1, 3, 8, 20, 50, 100):
            future = i + ahead
            if future >= len(nominal):
                continue
            if future not in targets:
                targets[future] = ContourResidual(nominal[future], geometry(future))
            for interior in (False, True):
                rings = _rings(nominal[future].region, interior)
                if not rings:
                    continue
                points = np.concatenate([_samples(r, 4.) for r in rings])
                if len(points) > 32:
                    points = points[np.linspace(0, len(points)-1, 32).astype(int)]
                values = np.stack([r(points, interior) for r in history])
                translations = np.stack([r.translation for r in history])
                xs.append(features(values, translations, ahead, interior))
                ys.append(targets[future](points, interior).astype('float32'))
    if capture_dir:
        import gzip
        capture_dir.mkdir(parents=True, exist_ok=True)
        frames = [dict(index=i, z=nominal[i].z, command=nominal[i].region.wkb_hex,
                       measured=measured_layers[i].region.wkb_hex if i in measured_layers else None,
                       actual=geometry(i).region.wkb_hex) for i in range(len(nominal))]
        with gzip.open(capture_dir / (identity + '.json.gz'), 'wt') as stream:
            json.dump(dict(group=identity, validation=validation, eligible=eligible.tolist(), frames=frames), stream)
    return np.concatenate(xs), np.concatenate(ys), dict(group=identity, validation=validation,
        frames=len(measured), samples=sum(len(x) for x in xs), seconds=time.time()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-seeds', nargs='+', type=int, default=list(range(60, 64)))
    parser.add_argument('--validation-seeds', nargs='+', type=int, default=[70, 71])
    parser.add_argument('--parts', nargs='+', default=['bracket', 'cup', 'mug', 'spacer'])
    parser.add_argument('--kinds', nargs='+', default=['clean', 'collapse', 'warp', 'shift', 'under_extrusion'])
    parser.add_argument('--snapshots', type=int, default=24)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--domain', choices=['synthetic', 'mujoco'], default='synthetic')
    parser.add_argument('--profiles', nargs='+', default=['cartesian'])
    parser.add_argument('--include-current', action='store_true', help='also train camera-to-current-deposit calibration')
    parser.add_argument('--save-trajectories', action='store_true', help='retain compact geometry for later training ablations')
    args = parser.parse_args()
    if set(args.train_seeds) & set(args.validation_seeds):
        parser.error('training and validation seeds must be disjoint')
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')):
        parser.error('dataset must be on the data volume')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'data.npz').exists():
        parser.error('refusing to overwrite a dataset')
    jobs = [(part, kind, seed, validation, args.snapshots, args.domain, profile, args.include_current,
             str(args.out / 'trajectories') if args.save_trajectories else None) for validation, seeds in
            [(False, args.train_seeds), (True, args.validation_seeds)]
            for profile in args.profiles for seed in seeds for part in args.parts for kind in args.kinds]
    manifest = dict(arguments={**vars(args), 'out': str(args.out)}, status='running',
        source_sha256=source_digest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        domain=args.domain, target='true future contour displacement',
        input='four past camera-derived contour residuals; horizon; boundary role', expected_groups=len(jobs))
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(trajectory, job) for job in jobs]
        for future in as_completed(futures):
            x, y, info = future.result()
            rows.append((x, y, info))
            print(json.dumps(info), flush=True)
    rows.sort(key=lambda row: row[2]['group'])
    x = np.concatenate([row[0] for row in rows]).astype('float32')
    y = np.concatenate([row[1] for row in rows]).astype('float32')
    validation = np.concatenate([np.full(len(a), info['validation'], dtype=bool) for a, _, info in rows])
    group = np.concatenate([np.full(len(a), info['group']) for a, _, info in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite training data')
    np.savez_compressed(args.out / 'data.npz', x=x, y=y, validation=validation, group=group)
    manifest.update(status='complete', rows=len(x), groups=[row[2] for row in rows],
                    final_source_sha256=source_digest())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print('Dataset complete:', len(x), 'samples', flush=True)


if __name__ == '__main__':
    main()
