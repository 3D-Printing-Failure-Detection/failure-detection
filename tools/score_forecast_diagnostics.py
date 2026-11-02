"""Score predictors on identical untouched camera trajectories.

One-step predictions precede the current measurement. Full-continuation
forecasts use only the prefix observed so far and are checked against the
unchanged functional spec; actual geometry is evaluation-only.
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
from recovery.geometry.spec import check
from recovery.eval.mujoco_eval import source_digest
from recovery.loop.contour import ContourObservationState
from recovery.loop.observation import ObservationState
from recovery.models.classical import boundary_deviation
from recovery.parts import get_part
from recovery.types import Layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results, inputs = [], {}
    initial_digest = source_digest()
    for path in sorted(args.directory.glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            record = json.load(stream)
        inputs[path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                capture_source_sha256=record['source_sha256'])
        part = get_part(record['part'])
        layers = {key: [Layer(f['z'], shapely.from_wkb(f[key])) for f in record['frames']]
                  for key in ('command', 'measured', 'actual')}
        for name, cls in [('original', ObservationState), ('contour', ContourObservationState)]:
            model = cls()
            errors, snapshots, measured = [], [], []
            for k, (command, observation, actual) in enumerate(zip(*layers.values())):
                if k >= record['onset'] and model.residuals:
                    forecast = model.predict([command])[0]
                    errors.append(boundary_deviation(forecast.region, actual.region))
                measured.append(model.observe(command, observation))
                # Short faults can end between sparse checkpoints. Include
                # every one of the first eight affected frames, then subsample.
                if k >= record['onset'] and (k - record['onset'] < 8 or (k - record['onset']) % args.stride == 0):
                    future = model.predict(layers['command'][k + 1:])
                    verdict = check(measured + future, layers['command'], part.spec, part.layer_h)
                    snapshots.append(dict(layer=k, predicted_ok=verdict.ok, violations=verdict.violations))
            row = dict(trajectory=path.name, model=name, actual_ok=record['final_ok'],
                       forecast_frames=len(errors), boundary_error_mm=float(np.mean(errors)),
                       continuation_snapshots=len(snapshots), predicted_failures=sum(not s['predicted_ok'] for s in snapshots),
                       snapshots=snapshots)
            results.append(row)
            print(path.name, name, 'actual', row['actual_ok'], 'forecast failure',
                  f"{row['predicted_failures']}/{len(snapshots)}", 'error', round(row['boundary_error_mm'], 3), flush=True)
    final_digest = source_digest()
    assert initial_digest == final_digest, 'source changed during scoring'
    manifest = dict(source_sha256=initial_digest, final_source_sha256=final_digest,
                    scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    stride=args.stride, dense_onset_frames=8, inputs=inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(manifest=manifest, scores=results), indent=2))


if __name__ == '__main__':
    main()
