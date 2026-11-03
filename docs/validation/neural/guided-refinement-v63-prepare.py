from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo = Path('/home/ishaan/failure-detection')
base = Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root, prior = base / 'neural-guided-refinement-v63', base / 'neural-native-coverage-v58'
source = root / 'source'
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
used = int(subprocess.check_output(['du', '-sx', '-B1', '/mnt/data/ishaan']).split()[0])
other_root = base / 'neural-observed-wall-v62'
other_used = int(subprocess.check_output(['du', '-sx', '-B1', str(other_root)]).split()[0])
other_reserved = max(0, 18_000_000 - other_used) + 1_000_000
assert used + 12_000_000 + other_reserved < 99_990_000_000, (used, other_reserved)
assert (prior / 'run.exit').read_text().strip() == '0' and read(prior / 'condition.json')['status'] == 'rejected'
old = read(prior / 'protocol.json')
root.mkdir()
source.mkdir()
for pattern in ['recovery/**/*.py', 'tools/*.py', 'parts/*.yaml']:
    for path in repo.glob(pattern):
        destination = source / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract
for origin, dest in [('/tmp/guided-refinement-v63-run.py', 'run.py'),
                     ('/tmp/guided-refinement-v63-screen.py', 'screen.py'), (__file__, 'prepare.py')]:
    shutil.copy2(origin, root / dest)
py = '/home/ishaan/failure-detection/.venv/bin/python'
penalties = dict(control=1., guided_linear=1., guided_nonlinear=1.)
training = {}
for name, penalty in penalties.items():
    command = old['training_commands']['linear'].copy()
    command[1] = str(source / 'tools/train_boundary_refinement.py')
    command[command.index('--out') + 1] = str(root / name)
    command[command.index('--overprediction-weight') + 1] = str(penalty)
    if name != 'control':
        command += ['--condition-on-images']
        command[command.index('--hidden-channels') + 1] = '8' if name == 'guided_nonlinear' else '0'
    training[name] = command
diagnostics = []
for oldjob in old['diagnostics']:
    name, case = oldjob['name'], oldjob['case']
    profile, part, kind, seed = case
    command = [py, str(source / 'tools/inspect_camera_case.py'), '--profile', profile, '--part', part,
               '--kind', kind, '--seed', str(seed), '--checkpoint', 'control=' + str(root / 'control/boundary.npz'),
               '--include-classical-base-hybrids', '--out', str(root / name)]
    diagnostics.append(dict(name=name, case=case, out=str(root / name), command_prefix=command))
collection = base / 'neural-context-camera-v8/trajectories/trajectories'
original = base / 'neural-boundary-v1/model/boundary.npz'
geometry = [py, str(source / 'tools/validate_camera_geometry.py'), '--collections', str(collection),
            '--validation-seeds', '70', '71', '72', '73', '400', '410', '417', '420', '--expected-groups', '36',
            '--checkpoint', 'original=' + str(original), '--checkpoint', 'control=' + str(root / 'control/boundary.npz'),
            '--workers', '2', '--out', str(root / 'geometry-validation')]
lost = sorted({tuple(row) for rows in read(prior / 'condition.json')['lost_original_true_detections'].values() for row in rows})
assert len(lost) == 3
p = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage='development', source=str(source),
         source_sha256=runtime_digest(source), simulation_contract_sha256=simulation_contract(source)['sha256'],
         purpose='Test image-conditioned native refinement after logit-only smoothing and asymmetric costs fail to distinguish valid thin material from real gaps',
         python=py, base_checkpoint=old['base_checkpoint'], base_checkpoint_sha256=old['base_checkpoint_sha256'],
         original_checkpoint=str(original), control_checkpoint_sha256=digest(prior / 'linear/boundary.npz'),
         augmentation_sha256=old['augmentation_sha256'], training_pythonpath=old['training_pythonpath'],
         penalties=penalties, candidates=['guided_linear', 'guided_nonlinear'], training_commands=training,
         trained_parameters=dict(control=10, guided_linear=37, guided_nonlinear=305),
         reference_root=str(prior), reference_geometry_frames=str(prior / 'geometry-validation/frames.jsonl'),
         collections=str(collection), good_onsets=old['identity_cases'], lost_detections=lost,
         diagnostics=diagnostics, geometry_command_prefix=geometry,
         training_protocol='Freeze base7cc weights. Reproduce fb422245 logit-only linear head exactly, then train a guided linear head with 37 parameters and a guided nonlinear residual head with 305 parameters. Each directly receives native logits plus the existing current grayscale,previous grayscale and current commanded mask repeated at parent-pixel coordinates. No new observation or target label enters inference. Retain 40 epochs, seed 2026, 3720 samples/epoch, 50/50 original/augmented sampling, and prior 50% critical-pixel loss normalization. Both guided heads start as identity. All overprediction weights are 1.',
         weights_selected_by='The unchanged unweighted distance objective on the same original 2040 validation tiles, at valid 62x62 coordinates. No critical-pixel reweighting of validation and no asymmetric cost. Only fixed selected checkpoints enter the consumed seven-frame screen.',
         screen_gate='Fixed selected checkpoints must reject all 3 unioned lost true detections and accept all 4 retained good onsets. Reproduce oldlinear/classical features on all 7 and original-camera features on the 3 re-rendered validation frames. Advance both independent camera noise streams through every preceding selected index. No image from fresh test seeds.',
         diagnostic_gate='Only screened heads advance to four full 160-layer nominal-command camera replays. All 640 truth,classical,oldlinear raw/hybrid features and finalverdicts must exactly reproduce v58. Each head must pass all 4 unchanged full specifications with existing classical specified bases.',
         geometry_gate='Only full-print-qualified heads advance to all 2040 original validation frames. Reproduce truth,classical,original and oldlinear features exactly; require every original true functional-defect detection preserved and total false negatives no higher. Report every component and false-positive count.',
         maximum_new_artifact_allowance_bytes=12_000_000, storage_bytes_at_launch=used, other_live_reserved_bytes=other_reserved,
         concurrent_reservation=dict(root=str(other_root), maximum_artifact_bytes=18_000_000, supervisor_log_bytes=1_000_000,
             policy='Conservatively reserve the remaining v62 allowance even if that run completes before this one'),
         reserved_fresh_seeds_untouched=[2001, 2002, 2003, 2004],
         limitation='Camera qualification only. Rejected heads cannot enter recovery. A qualified head still needs a new actual neural/Qwen recovery protocol and full historical,development and fresh baseline comparisons; no automatic promotion or final evaluation.')
pins = [root / name for name in ['run.py', 'screen.py', 'prepare.py']]
# Pin the same input artifacts as the prior training plus all new reference controls.
for path in old['pins']:
    path = Path(path)
    if path.is_relative_to(Path(old['source'])):
        pins.append(source / path.relative_to(Path(old['source'])))
    elif path != prior / 'run.py':
        pins.append(path)
pins += [source / 'tools/experiment_provenance.py', prior / 'protocol.json', prior / 'condition.json',
         prior / 'linear/boundary.npz', prior / 'linear/training.json', Path(p['reference_geometry_frames']),
         prior / 'geometry-validation/manifest.json', collection.parent / 'manifest.json']
for job in diagnostics:
    pins += [prior / job['name'] / name for name in ['manifest.json', 'layers.json', 'final.json']]
for path in collection.glob('*.json.gz'):
    # All retained validation inputs are immutable, including fast-screen frames.
    import gzip
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    if record['validation'] and record['group'].split('-')[1] in ['cup', 'mug']:
        pins.append(path)
pins += [other_root / name for name in ['protocol.json', 'run.py', 'launch.json']]
pins += [base / 'neural-material-penalty-v61' / name for name in ['condition.json', 'screen.json', 'run.exit']]
p['pins'] = {str(path): digest(path) for path in pins}
assert p['source_sha256'] == runtime_digest(repo)
assert p['simulation_contract_sha256'] == 'fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n')
launch = '#!/bin/bash\nset -uo pipefail\ncd ' + shlex.quote(str(source)) + '\n'
launch += 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch += shlex.join([py, '-u', str(root / 'run.py')]) + ' > ' + shlex.quote(str(root / 'run.log')) + ' 2>&1\n'
launch += 'result=$?\nprintf "%s\\n" "$result" > ' + shlex.quote(str(root / 'run.exit')) + '\nexit "$result"\n'
(root / 'launch.sh').write_text(launch)
window = subprocess.check_output(['tmux', 'new-window', '-d', '-t', 'neural-recovery', '-n', 'guided-refinement-v63',
                                  '-P', '-F', '#{window_id}', 'bash ' + shlex.quote(str(root / 'launch.sh'))], text=True).strip()
(root / 'launch.json').write_text(json.dumps(dict(tmux_window=window, session='neural-recovery',
                                                protocol_sha256=digest(root / 'protocol.json'), runner_sha256=digest(root / 'run.py')),
                                           indent=2) + '\n')
print(json.dumps(dict(root=str(root), window=window, used=used, source_sha256=p['source_sha256'])))
