from pathlib import Path
import gzip
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = read(root/'protocol.json'); source = Path(p['source'])
sys.path.insert(0, str(source))
import cv2
import numpy as np
import shapely
from recovery import camera
from recovery.deform import random_failure
from recovery.eval.camera_trace import camera_frame_digest
from recovery.geometry import features as F
from recovery.geometry.ops import clean
from recovery.geometry.spec import OVERHANG_AREA
from recovery.models.boundary import BoundarySegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.experiment_provenance import runtime_digest, simulation_contract
from tools.inspect_camera_case import layer_features
from tools.validate_camera_geometry import frame_indices

flags = ('open_wall', 'hard_thin', 'base_problem', 'connection_problem')
fatal = lambda features: any(features.get(k, False) for k in flags)
models = {entry['name']: BoundarySegmenter(entry['checkpoint']) for entry in p['models']}
nonlinear = next(entry['name'] for entry in p['models'] if entry['sha256'].startswith('24400f10'))
rows, support = [], []


def verify():
    assert runtime_digest(source) == p['source_sha256']
    assert simulation_contract(source)['sha256'] == p['simulation_contract_sha256']
    for path, expected in p['pins'].items():
        assert digest(path) == expected, path


def inspect(image, previous, command, previous_command, actual, part, reference):
    truth = layer_features(actual, command, part.spec)
    measured, layers = {}, {}
    expected, previous_expected = camera.expected_mask(command), camera.expected_mask(previous_command)
    for name, model in models.items():
        model.prime_history(previous, previous_expected)
        region = clean(model.measure(image, expected).simplify(.25/4))
        layers[name] = Layer(command.z, region)
        measured[name] = layer_features(layers[name], command, part.spec)
    for actual_features, old in [(truth, reference['truth']), (measured[nonlinear], reference['measured']['nonlinear'])]:
        for key, value in old.items():
            if key not in actual_features:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert abs(actual_features[key]-value) < 1e-7, (key, actual_features[key], value)
            else:
                assert actual_features[key] == value
    return dict(truth=truth, measured=measured, image_sha256_128=camera_frame_digest(image).hex(),
        previous_image_sha256_128=camera_frame_digest(previous).hex(),
        current_command_sha256=digest_bytes(command.region.wkb), previous_command_sha256=digest_bytes(previous_command.region.wkb)), layers


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def save_rows():
    (root/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


verify()
(root/'condition.json').write_text(json.dumps(dict(status='running', models=len(models), completed_frames=0))+'\n')
for case in p['consumed_cases']:
    print('START', case['name'], flush=True)
    part = get_part(case['part']); nominal = part.layers()
    refs = {row['layer']: row for row in read(case['reference_frames'])}
    failure = random_failure(nominal, np.random.default_rng(case['seed']), kind='under_extrusion')
    selected = set(case['layers']); histories = {name:{} for name in models}
    with MujocoPrinter(nominal, failure, case['seed'], case['profile'], False) as printer:
        previous = None
        for index in range(max(selected)+1):
            actual = printer.print_layer(); image = printer.camera_image()
            if index in selected:
                row, layers = inspect(image, image if previous is None else previous, nominal[index],
                    nominal[max(0,index-1)], actual, part, refs[index])
                row.update(case=case['name'], layer=index, split='consumed_false_alarm')
                rows.append(row)
                for name, layer in layers.items():
                    histories[name][index] = layer
                if index == sorted(selected)[1]:
                    for label, value in [('current', image), ('previous', previous)]:
                        assert cv2.imwrite(str(root/(case['name']+'-'+label+'.png')), value)
                print('FRAME', case['name'], index, 'nonfatal_models', [name for name in models if not fatal(row['measured'][name])], flush=True)
                save_rows()
            previous = image
    pairs = []
    for index in sorted(selected):
        if index-1 in selected:
            pairs.append(dict(layer=index, excess_overhang_mm2={name:
                F.excess_overhang(history[index], history[index-1], nominal[index], nominal[index-1])
                for name, history in histories.items()}))
    support.append(dict(case=case['name'], pairs=pairs))
    (root/'support.json').write_text(json.dumps(support, indent=2)+'\n')

validation = [json.loads(line) for line in Path(p['validation_frames']).read_text().splitlines()]
for job in p['real_defect_groups']:
    with gzip.open(job['trajectory'], 'rt') as stream:
        record = json.load(stream)
    assert record['validation']
    part = get_part(job['part']); nominal = part.layers()
    actual = [Layer(frame['z'], shapely.from_wkb(frame['actual'])) for frame in record['frames']]
    commands = [Layer(frame['z'], shapely.from_wkb(frame['command'])) for frame in record['frames']]
    assert all(a.region.equals(b.region) and a.z == b.z for a,b in zip(commands,nominal))
    indices = frame_indices(record, part, 12)
    for profile, selected in job['profile_layers'].items():
        previous_seed = int.from_bytes(hashlib.sha256((record['group']+profile+'-previous-camera').encode()).digest()[:8], 'little')
        with MujocoPrinter(nominal,None,420,profile,False) as printer, MujocoPrinter(nominal,None,previous_seed,profile,False) as previous_printer:
            for index in indices:
                printer.printed=actual[:index+1]; image=printer.camera_image(); previous=image
                if index > 0:
                    previous_printer.printed=actual[:index]; previous=previous_printer.camera_image()
                if index not in selected:
                    continue
                ref=next(row for row in validation if (row['group'],row['profile'],row['layer'])==(record['group'],profile,index))
                row, _ = inspect(image, previous, commands[index], commands[max(index-1,0)], actual[index], part, ref)
                row.update(case=record['group']+':'+profile, layer=index, split='selected_real_defect')
                assert fatal(row['truth'])
                rows.append(row); save_rows()
                print('REAL', row['case'],index, 'missed', [name for name in models if not fatal(row['measured'][name])], flush=True)

verify()
assert len(rows) == 22
summary = {}
for name in models:
    real = [row for row in rows if row['split'] == 'selected_real_defect']
    cases = {}
    for case in p['consumed_cases']:
        selected = [row for row in rows if row['case'] == case['name']]
        pair = next(item for item in support if item['case']==case['name'])
        maximum = max(item['excess_overhang_mm2'][name] for item in pair['pairs'])
        cases[case['name']] = dict(no_false_fatal_geometry=not any(fatal(row['measured'][name]) for row in selected),
                                  max_local_overhang_mm2=maximum,
                                  local_gate_passes=not any(fatal(row['measured'][name]) for row in selected) and maximum <= OVERHANG_AREA)
    summary[name] = dict(selected_true_detections=sum(fatal(row['measured'][name]) for row in real), cases=cases)
coverage = {case['name']: [name for name, result in summary.items() if result['selected_true_detections']==6
                         and result['cases'][case['name']]['local_gate_passes']] for case in p['consumed_cases']}
result = dict(status='complete', models=len(models), frames=len(rows), predictions=len(models)*len(rows),
    reference_predictions_and_truth_reproduced=True, models_summary=summary,
    local_case_coverage_preserving_six_selected_defects=coverage,
    all_four_locally_covered=all(coverage.values()), protocol_sha256=digest(root/'protocol.json'),
    frames_sha256=digest(root/'frames.jsonl'), support_sha256=digest(root/'support.json'),
    new_training_runs=0, new_recovery_trials=0, limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'condition.json').write_text(json.dumps(dict(status='complete', models=len(models),completed_frames=len(rows),
    all_four_locally_covered=all(coverage.values()),comparison_sha256=digest(root/'comparison.json')),indent=2)+'\n')
print(json.dumps(dict(status='complete',coverage=coverage)),flush=True)
