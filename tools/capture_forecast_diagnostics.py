"""Capture untouched camera trajectories for offline, same-frame forecasts.

Simulator truth is stored only for evaluation, never passed to an estimator.
No images are retained. Use a data-volume output directory.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.deform import random_failure
from recovery.eval.mujoco_eval import source_digest
from recovery.geometry.spec import check
from recovery.models.classical import ClassicalSegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', nargs='+', default=['cup', 'mug'])
    parser.add_argument('--kinds', nargs='+', default=['warp', 'collapse', 'under_extrusion', 'shift'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--profile', default='cartesian')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.parts:
        for kind in args.kinds:
            path = args.out / f'{args.profile}-{name}-{kind}-{args.seed}.json.gz'
            if path.exists():
                raise ValueError(f'refusing to overwrite {path}')
            part = get_part(name)
            nominal = part.layers()
            failure = random_failure(nominal, np.random.default_rng(args.seed), kind=kind)
            segmenter = ClassicalSegmenter()
            frames = []
            with MujocoPrinter(nominal, failure, args.seed, args.profile, False) as printer:
                for layer in nominal:
                    actual = printer.print_layer()
                    measured = segmenter.measure(printer.camera_image())
                    frames.append(dict(z=layer.z, command=layer.region.wkb_hex,
                                       measured=measured.wkb_hex, actual=actual.region.wkb_hex))
                verdict = check(printer.printed, nominal, part.spec, part.layer_h)
            record = dict(part=name, kind=kind, profile=args.profile, seed=args.seed,
                          onset=failure.k0, final_ok=verdict.ok, violations=verdict.violations,
                          source_sha256=source_digest(), frames=frames)
            with gzip.open(path, 'wt') as stream:
                json.dump(record, stream)
            print(path.name, 'functional', verdict.ok, 'onset', failure.k0, flush=True)


if __name__ == '__main__':
    main()
