"""Inspect camera errors on a nominal-command simulation, without recovery.

Truth is used only for diagnostic scoring. Each segmenter receives exactly the
same actual camera frame and expected mask as it would in the controller. This
is not a recovery success benchmark or an input to any runtime policy.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.deform import KINDS, random_failure
from recovery.eval.mujoco_eval import source_digest
from recovery.geometry import features as F
from recovery.geometry.ops import clean, iou
from recovery.geometry.spec import HARD_WALL_FRAC, check
from recovery.models.boundary import BoundarySegmenter
from recovery.models.classical import ClassicalSegmenter, boundary_deviation
from recovery.parts import PART_NAMES, get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.printer.profiles import PROFILES
from recovery.types import Layer


def layer_features(layer, nominal, spec):
    base = F.base(layer)
    row = dict(area_mm2=base['area'], components=base['n'],
               hole_area_mm2=base['hole_area'], main_holes=base['n_main_holes'])
    container = spec.get('container')
    if container and layer.z > container['base_h']:
        row['open_wall'] = base['hole_area'] < .5 * F.base(nominal)['hole_area']
    thickness = spec.get('min_wall_mm')
    if thickness:
        for name, size in [('soft_thin', thickness), ('hard_thin', HARD_WALL_FRAC * thickness)]:
            excess = F.thin_frac(layer, size) - F.thin_frac(nominal, size)
            row[name + '_excess'] = excess
            row[name] = excess > .03
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='cartesian')
    parser.add_argument('--part', choices=PART_NAMES, required=True)
    parser.add_argument('--kind', choices=['clean', *KINDS], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--snapshot-layers', nargs='*', type=int, default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')):
        parser.error('diagnostic artifacts must stay on the data volume')
    if args.out.exists():
        parser.error('refusing to overwrite an existing diagnostic')
    segmenters = {'classical': ClassicalSegmenter()}
    checkpoint_paths = {}
    for item in args.checkpoint:
        label, separator, path = item.partition('=')
        if not separator or not label or label in segmenters:
            parser.error('checkpoints need unique LABEL=PATH entries')
        segmenters[label] = BoundarySegmenter(path)
        checkpoint_paths[label] = str(Path(path).resolve())
    part = get_part(args.part)
    nominal = part.layers()
    if any(i < 0 or i >= len(nominal) for i in args.snapshot_layers):
        parser.error('snapshot index is outside the print')
    failure = None if args.kind == 'clean' else random_failure(nominal, np.random.default_rng(args.seed), kind=args.kind)
    args.out.mkdir(parents=True)
    manifest = dict(status='running', source_sha256=source_digest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256={name: segmenters[name].sha256 for name in checkpoint_paths},
        follows_nominal_commands=True, recovery_actions=0,
        limitation='diagnostic measurements on a fixed trajectory; not closed-loop recovery evidence')
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    histories = {name: [] for name in segmenters}
    rows = []
    with MujocoPrinter(nominal, failure, args.seed, args.profile, False) as printer:
        for index, command in enumerate(nominal):
            actual = printer.print_layer()
            image = printer.camera_image()
            expected_mask = camera.expected_mask(command)
            row = dict(layer=index, z=command.z, truth=layer_features(actual, command, part.spec), measured={})
            for name, segmenter in segmenters.items():
                region = clean(segmenter.measure(image, expected_mask).simplify(.25 / 4))
                measured = Layer(command.z, region)
                histories[name].append(measured)
                row['measured'][name] = dict(layer_features(measured, command, part.spec),
                    iou=iou(region, actual.region), boundary_error_mm=boundary_deviation(region, actual.region))
            rows.append(row)
            if index in args.snapshot_layers:
                if not cv2.imwrite(str(args.out / f'camera-{index:03d}.png'), image):
                    raise RuntimeError('camera snapshot was not written')
                geometry = dict(nominal=command.region.wkb_hex, truth=actual.region.wkb_hex,
                    measured={name: history[-1].region.wkb_hex for name, history in histories.items()})
                (args.out / f'geometry-{index:03d}.json').write_text(json.dumps(geometry))
            if (index + 1) % 20 == 0:
                print(f'{index + 1}/{len(nominal)} actual frames inspected', flush=True)
        final = dict(truth=asdict(check(printer.printed, nominal, part.spec, part.layer_h)),
            measured={name: asdict(check(history, nominal, part.spec, part.layer_h))
                      for name, history in histories.items()})
        manifest.update(status='complete', frames=printer.camera_frames,
                        layers_printed=printer.n_done, final_source_sha256=source_digest())
    (args.out / 'layers.json').write_text(json.dumps(rows, indent=2))
    (args.out / 'final.json').write_text(json.dumps(final, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
