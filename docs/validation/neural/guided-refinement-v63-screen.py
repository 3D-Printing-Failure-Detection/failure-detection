"""Screen fixed trained cameras on seven consumed, exactly reproduced frames."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = read(root / 'protocol.json')
sys.path.insert(0, p['source'])
import cv2
import numpy as np
import shapely
from recovery import camera
from recovery.geometry.ops import clean
from recovery.models.boundary import BoundarySegmenter
from recovery.models.classical import ClassicalSegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.inspect_camera_case import layer_features
from tools.validate_camera_geometry import frame_indices
from tools.experiment_provenance import runtime_digest

assert runtime_digest(p['source']) == p['source_sha256']
models = {name: BoundarySegmenter(root / name / 'boundary.npz')
          for name in ['control', *p['candidates']]}
models['original'] = BoundarySegmenter(p['original_checkpoint'])
models['classical'] = ClassicalSegmenter()
flags = ['open_wall', 'hard_thin', 'base_problem', 'connection_problem']
fatal = lambda value: any(value.get(key, False) for key in flags)
rows = []


def measure(image, previous, part, index):
    nominal = part.layers()
    expected = camera.expected_mask(nominal[index])
    values = {}
    for name, model in models.items():
        if getattr(model, 'input_frames', 1) == 2:
            model.prime_history(previous, camera.expected_mask(nominal[max(index - 1, 0)]))
        region = clean(model.measure(image, expected).simplify(.25 / 4))
        values[name] = layer_features(Layer(nominal[index].z, region), nominal[index], part.spec)
    return values


for case in p['good_onsets']:
    image, previous = cv2.imread(case['image']), cv2.imread(case['previous'])
    assert image is not None and previous is not None
    part = get_part(case['part'])
    reference = read(Path(p['reference_root']) / case['name'] / 'layers.json')[case['layer']]
    measured = measure(image, previous, part, case['layer'])
    for name, label in [('control', 'linear'), ('classical', 'classical')]:
        expected = {k: v for k, v in reference['measured'][label].items()
                    if k not in ('iou', 'boundary_error_mm')}
        assert measured[name] == expected, (case['name'], name, 'onset control mismatch')
    assert not fatal(reference['truth'])
    rows.append(dict(case=case['name'], layer=case['layer'], desired_fatal=False,
                     truth=reference['truth'], measured=measured,
                     current_image_sha256=digest(case['image']), previous_image_sha256=digest(case['previous']),
                     current_array_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                     previous_array_sha256=hashlib.sha256(previous.tobytes()).hexdigest()))

reference = {(r['group'], r['profile'], r['layer']): r for r in
             map(json.loads, Path(p['reference_geometry_frames']).read_text().splitlines())}
snapshots = root / 'screen-images'
snapshots.mkdir()
for target in p['lost_detections']:
    group, profile, target_index = target
    path = Path(p['collections']) / (group + '.json.gz')
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    assert record['validation'] and record['group'] == group
    _, part_name, _, seed = group.split('-')
    part = get_part(part_name)
    nominal = part.layers()
    actual = [Layer(frame['z'], shapely.from_wkb(frame['actual'])) for frame in record['frames']]
    assert len(actual) == len(nominal)
    assert all(frame['z'] == layer.z and shapely.from_wkb(frame['command']).equals_exact(layer.region, 1e-8)
               for frame, layer in zip(record['frames'], nominal))
    indices = frame_indices(record, part, 12)
    assert target_index in indices
    previous_seed = int.from_bytes(hashlib.sha256((group + profile + '-previous-camera').encode()).digest()[:8], 'little')
    rendered = []
    with MujocoPrinter(nominal, None, int(seed), profile, False) as printer, \
            MujocoPrinter(nominal, None, previous_seed, profile, False) as previous_printer:
        for index in indices:
            if index > target_index:
                break
            printer.printed = actual[:index + 1]
            image = printer.camera_image()
            previous = image
            if index > 0:
                previous_printer.printed = actual[:index]
                previous = previous_printer.camera_image()
            rendered.append(index)
            if index != target_index:
                continue
            truth = layer_features(actual[index], nominal[index], part.spec)
            measured = measure(image, previous, part, index)
            ref = reference[(group, profile, index)]
            assert truth == ref['truth'] and fatal(truth) and fatal(ref['measured']['original'])
            for name, label in [('control', 'linear'), ('original', 'original'), ('classical', 'classical')]:
                assert measured[name] == ref['measured'][label], (target, name, 'validation control mismatch')
            files = {}
            for label, value in [('current', image), ('previous', previous)]:
                dest = snapshots / f'{group}-{profile}-{index}-{label}.png'
                assert cv2.imwrite(str(dest), value)
                assert np.array_equal(cv2.imread(str(dest)), value), 'PNG round trip changed camera input'
                files[label] = dict(path=str(dest), sha256=digest(dest),
                                    array_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            rows.append(dict(case=group + '/' + profile, layer=index, desired_fatal=True,
                             truth=truth, measured=measured, rendered_indices=rendered.copy(),
                             trajectory_sha256=digest(path), images=files))

assert len(rows) == 7
passes = {name: all(fatal(row['measured'][name]) == row['desired_fatal'] for row in rows)
          for name in p['candidates']}
result = dict(status='complete', source_sha256=p['source_sha256'], script_sha256=digest(__file__),
              protocol_sha256=digest(root / 'protocol.json'), frames=len(rows),
              checkpoint_sha256={name: model.sha256 for name, model in models.items() if name != 'classical'},
              all_seven_control_feature_records_reproduced=True, passes=passes, rows=rows,
              limitation='Fixed-checkpoint screen on consumed development frames; not a new validation set, full-print qualification or recovery rate.')
(root / 'screen.json').write_text(json.dumps(result, indent=2) + '\n')
for row in rows:
    print(json.dumps(dict(case=row['case'], layer=row['layer'], desired_fatal=row['desired_fatal'],
                          predicted={name: fatal(value) for name, value in row['measured'].items()})), flush=True)
print(json.dumps(dict(screen_passes=passes)), flush=True)
