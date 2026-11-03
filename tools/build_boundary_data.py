"""Train camera boundaries from re-rendered, retained MuJoCo deposited layers.

Only simulator truth supplies labels and sampling locations. Runtime features
are grayscale patches, with no nominal shape, fault identity or truth input.
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

import cv2
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.eval.mujoco_eval import source_digest
from recovery.models.boundary import PATCH_SIZE, normalized_gray, subpixel_offsets, camera_patches
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.boundary_tiles import tile_inputs, tile_query_pixels


def build_case(job):
    path, n_frames, n_pixels = job[:3]
    all_profiles, dense_start = job[3:5] if len(job) > 3 else (False, False)
    output_upsample = job[5] if len(job) > 5 else 1
    include_base, focus_deformed = job[6:8] if len(job) > 6 else (False, False)
    input_scales = job[8] if len(job) > 8 else (1,)
    input_frames = job[9] if len(job) > 9 else 1
    tile_size = job[10] if len(job) > 10 else 0
    offsets = subpixel_offsets(output_upsample)
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    _, name, _, seed = record['group'].split('-')
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(record['group'].encode()).digest()[:8], 'little'))
    actual = [Layer(f['z'], shapely.from_wkb(f['actual'])) for f in record['frames']]
    part = get_part(name)
    nominal = part.layers()
    eligible = record['eligible']
    count = min(n_frames, len(eligible))
    if dense_start:
        dense_count = min(10, count)
        positions = list(range(dense_count))
        if count > dense_count:
            positions += np.linspace(dense_count, len(eligible)-1, count-dense_count).astype(int).tolist()
        indices = np.asarray(eligible)[positions]
    else:
        indices = np.array(eligible)[np.linspace(0, len(eligible)-1, count).astype(int)]
    if include_base and part.spec.get('container'):
        base_indices = [i for i, layer in enumerate(nominal) if layer.z <= part.spec['container']['base_h']]
        indices = np.array(sorted(set(indices.tolist()) | set(base_indices)))
    xs, ys = [], []
    focused_samples = 0
    previous_frames = 0
    # Camera-profile augmentation reuses the deposited geometry. It does not
    # pretend the same toolpath was physically re-executed by every printer.
    profiles = ['cartesian', 'corexy', 'bedslinger']
    for offset, profile in enumerate(profiles):
        selected = indices if all_profiles else indices[offset::len(profiles)]
        if not len(selected):
            continue
        # An independent renderer keeps the existing current-frame pixels and
        # noise sequence identical. The prior image is the same camera view of
        # the actual preceding deposit, with an independent camera-noise stream.
        previous_seed = int.from_bytes(hashlib.sha256(
            (record['group'] + profile + '-previous-camera').encode()).digest()[:8], 'little')
        previous_context = (MujocoPrinter(nominal, None, previous_seed, profile, False)
                            if input_frames == 2 else nullcontext(None))
        with MujocoPrinter(nominal, None, int(seed), profile, False) as printer, previous_context as previous_printer:
            for index in selected:
                printer.printed = actual[:index+1]
                gray = normalized_gray(printer.camera_image())
                layer = actual[index]
                mask = camera.expected_mask(layer)
                border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((9, 9), 'uint8'))
                y, x = np.nonzero(border)
                if not len(y):
                    continue
                focus_pool = np.empty(0, dtype=int)
                if focus_deformed:
                    difference = (mask != camera.expected_mask(nominal[index])).astype('uint8')
                    neighborhood = cv2.dilate(difference, np.ones((9, 9), 'uint8'))
                    focus_pool = np.flatnonzero(neighborhood[y, x])
                if tile_size:
                    n_focus = int(bool(len(focus_pool)) and rng.random() < .5)
                    chosen = rng.choice(focus_pool if n_focus else np.arange(len(y)))
                    origin = (int(np.clip(x[chosen]-tile_size//2, 0, gray.shape[1]-tile_size)),
                              int(np.clip(y[chosen]-tile_size//2, 0, gray.shape[0]-tile_size)))
                else:
                    n_focus = n_pixels // 2 if len(focus_pool) else 0
                    chosen = np.r_[rng.choice(focus_pool, n_focus, replace=len(focus_pool) < n_focus),
                                   rng.choice(len(y), n_pixels-n_focus, replace=len(y) < n_pixels-n_focus)]
                    yy = np.r_[y[chosen], rng.integers(PATCH_SIZE, gray.shape[0]-PATCH_SIZE, n_pixels // 4)]
                    xx = np.r_[x[chosen], rng.integers(PATCH_SIZE, gray.shape[1]-PATCH_SIZE, n_pixels // 4)]
                focused_samples += n_focus
                frames = [gray]
                if input_frames == 2:
                    previous_gray = gray
                    if index > 0:
                        previous_printer.printed = actual[:index]
                        previous_gray = normalized_gray(previous_printer.camera_image())
                        previous_frames += 1
                    frames.append(previous_gray)
                if tile_size:
                    patches = tile_inputs(frames, origin, tile_size)[None]
                    query_pixels = tile_query_pixels(origin, tile_size, output_upsample)
                else:
                    patches = np.concatenate([camera_patches(frame, np.column_stack([xx, yy]), input_scales)
                                              for frame in frames], axis=1)
                    query_pixels = np.column_stack([xx, yy])[:, None, :] + offsets[None, :, :]
                points = camera.px_to_mm(query_pixels.reshape(-1, 2))
                inside = shapely.contains_xy(layer.region, points[:, 0], points[:, 1])
                distance = shapely.distance(layer.region.boundary, shapely.points(points))
                target = np.clip(distance * np.where(inside, 1, -1), -1, 1).astype('float32')
                if tile_size:
                    target = target.reshape(1, 1, tile_size*output_upsample, tile_size*output_upsample).astype('float16')
                elif output_upsample > 1:
                    target = target.reshape(-1, output_upsample**2)
                xs.append(patches.astype('float16'))
                ys.append(target)
    return np.concatenate(xs), np.concatenate(ys), dict(group=record['group'],
        validation=record['validation'], frames=len(indices)*(len(profiles) if all_profiles else 1),
        previous_camera_images=previous_frames,
        focused_boundary_samples=focused_samples,
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--pixels', type=int, default=320)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--all-profiles', action='store_true', help='render every selected frame with all three cameras')
    parser.add_argument('--dense-start', action='store_true', help='retain the first ten eligible frames before subsampling')
    parser.add_argument('--output-upsample', type=int, choices=[1, 2, 4], default=1,
                        help='label distinct subpixel positions for native high-resolution prediction')
    parser.add_argument('--include-base-frames', action='store_true', help='include every solid container base layer')
    parser.add_argument('--focus-deformed', action='store_true',
                        help='sample half of boundary patches near deposited/command differences; training sampling only')
    parser.add_argument('--input-scales', type=int, nargs='+', choices=[1, 2, 4], default=[1],
                        help='pixel strides for each 9x9 image patch; 1 4 adds wider visual context')
    parser.add_argument('--input-frames', type=int, choices=[1, 2], default=1,
                        help='two includes the actual preceding layer image; first layer duplicates the current image')
    parser.add_argument('--dense-tile-size', type=int, choices=[0, 16], default=0,
                        help='one dense tile per selected frame; requires --pixels 1, --input-scales 1 and native 4x outputs')
    args = parser.parse_args()
    if len(set(args.input_scales)) != len(args.input_scales):
        parser.error('input scales must be distinct')
    if args.dense_tile_size and (args.input_scales != [1] or args.output_upsample != 4 or args.pixels != 1):
        parser.error('dense tiles require --pixels 1 --input-scales 1 --output-upsample 4')
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')):
        parser.error('training data must stay on the data volume')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'data.npz').exists():
        parser.error('refusing to overwrite a dataset')
    paths = sorted(args.trajectories.glob('*.json.gz'))
    if not paths:
        parser.error('no completed retained trajectories')
    capture_manifest = json.loads((args.trajectories.parent / 'manifest.json').read_text())
    if capture_manifest['status'] != 'complete' or len(paths) != capture_manifest['expected_groups']:
        parser.error('source trajectory collection is incomplete')
    manifest = dict(status='running', source_sha256=source_digest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        domain='MuJoCo renders of retained deposited geometry, with three camera profiles',
        input=f'normalized 9x9 grayscale patches at strides {args.input_scales}, current-first with {args.input_frames} frame(s)',
        previous_image_noise='independent deterministic camera stream' if args.input_frames == 2 else None,
        target='true signed distance in mm clipped to [-1,1]',
        output_upsample=args.output_upsample, output_offsets_xy_px=subpixel_offsets(args.output_upsample).tolist())
    if args.dense_tile_size:
        manifest.update(input='normalized current-first image tiles, including the valid-convolution border',
                        dense_tile_size=args.dense_tile_size, target_storage='float16, clipped signed distance in mm',
                        tile_helper_sha256=hashlib.sha256(Path(__file__).with_name('boundary_tiles.py').read_bytes()).hexdigest())
    manifest['capture_manifest_sha256'] = hashlib.sha256((args.trajectories.parent / 'manifest.json').read_bytes()).hexdigest()
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(build_case, (p, args.frames, args.pixels, args.all_profiles, args.dense_start,
                                               args.output_upsample, args.include_base_frames, args.focus_deformed,
                                               args.input_scales, args.input_frames, args.dense_tile_size)) for p in paths]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row[2]), flush=True)
    rows.sort(key=lambda row: row[2]['group'])
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    validation = np.concatenate([np.full(len(x), info['validation'], dtype=bool) for x, _, info in rows])
    group = np.concatenate([np.full(len(x), info['group']) for x, _, info in rows])
    np.savez_compressed(args.out / 'data.npz', x=x, y=y, validation=validation, group=group,
                        output_upsample=np.array(args.output_upsample), input_scales=np.array(args.input_scales),
                        input_frames=np.array(args.input_frames), dense_tile_size=np.array(args.dense_tile_size))
    manifest.update(status='complete', groups=[row[2] for row in rows], rows=len(x),
                    final_source_sha256=source_digest())
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Boundary dataset complete:', len(x), 'patches', flush=True)


if __name__ == '__main__':
    main()
