from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import urllib.request

repo = Path('/home/ishaan/failure-detection')
base = Path('/mnt/data/ishaan/hri-workspace/failure-detection')
previous = base / 'neural-continuation-fallback-v51'
root = base / 'neural-observed-wall-v62'
source = root / 'source'
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
used = int(subprocess.check_output(['du', '-sx', '-B1', '/mnt/data/ishaan']).split()[0])
assert used + 19_000_000 < 99_990_000_000, used
assert (previous / 'run.exit').read_text().strip() == '0' and read(previous / 'condition.json')['status'] == 'passed'
assert (base / 'neural-material-penalty-v61/run.exit').read_text().strip() == '0'
old = read(previous / 'protocol.json')
supervisor_path = base / 'local-supervisor-worker-v1/manifest.json'
supervisor = read(supervisor_path)
actual = Path('/proc/2524739/cmdline').read_bytes().decode().strip('\0').split('\0')
assert actual == supervisor['server_command']
with Path(supervisor['reuses_model_file']).open('rb') as stream:
    model_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert model_hash == supervisor['verified_model_sha256']
with urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=10) as response:
    health = json.load(response)
assert health['status'] == 'ok'
root.mkdir()
source.mkdir()
for pattern in ['recovery/**/*.py', 'tools/*.py', 'parts/*.yaml']:
    for path in repo.glob(pattern):
        destination = source / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract
shutil.copy2('/tmp/observed-wall-v62-run.py', root / 'run.py')
shutil.copy2(__file__, root / 'prepare.py')
attestation = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=2524739,
                   server_command=actual, health=health, model_sha256=model_hash, manifest_sha256=digest(supervisor_path))
(root / 'supervisor-runtime.json').write_text(json.dumps(attestation, indent=2) + '\n')
common = old['paired_jobs'][0]['command']
common = common[:common.index('--methods')] + ['--boundary-trend-continuation-fallback']
target_jobs = []
for name, enabled in [('control-defect', False), ('candidate-defect', True)]:
    command = common.copy()
    if enabled:
        command += ['--boundary-observed-wall-guard']
    command += ['--methods', *old['methods'], '--n', '1', '--workers', '2', '--profiles', 'bedslinger',
                '--parts', 'mug', '--kinds', 'under_extrusion', '--seed', '1004', '--out', str(root / name / 'results')]
    target_jobs.append(dict(name=name, enabled=enabled, command=command, expected_rows=2))
reference_jobs = [job for job in old['paired_jobs'] if job['enabled']] + old['regression_jobs']
reference_directories = [str(previous / job['name'] / 'results') for job in reference_jobs]
preservation_jobs = []
for job in reference_jobs:
    command = job['command'].copy()
    command[command.index('--out') + 1] = str(root / job['name'] / 'results')
    assert '--boundary-trend-continuation-fallback' in command
    command += ['--boundary-observed-wall-guard']
    preservation_jobs.append(dict(name=job['name'], enabled=True, command=command, expected_rows=job['expected_rows']))
neural = base / 'neural-equivariant-v1/classical/temporal.npz'
boundary = base / 'neural-boundary-v1/model/boundary.npz'
pins = [root / 'run.py', root / 'prepare.py', root / 'supervisor-runtime.json', supervisor_path,
        previous / 'protocol.json', previous / 'condition.json', previous / 'comparison.json',
        base / 'neural-known-defect-v60/comparison.json', base / 'neural-native-coverage-v58/condition.json',
        neural, neural.parent / 'training.json', boundary, boundary.parent / 'training.json']
for directory in map(Path, reference_directories):
    pins += [directory / 'manifest.json', directory.parent / 'audit/audit.json']
    pins += [directory / 'runs.jsonl'] if (directory / 'runs.jsonl').exists() else [directory / 'runs.jsonl.gz', directory / 'runs-storage.json']
pins += [source / 'tools' / name for name in ['audit_neural_experiment.py', 'experiment_provenance.py', 'result_storage.py']]
p = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage='development', source=str(source),
         source_sha256=runtime_digest(source), simulation_contract_sha256=simulation_contract(source)['sha256'],
         methods=old['methods'], target_jobs=target_jobs, target_case=['bedslinger', 'mug', 'under_extrusion', 1004],
         target_hard_wall_layer=101, preservation_jobs=preservation_jobs,
         preservation_cases=old['paired_cases'] + old['regression_cases'], reference_directories=reference_directories,
         neural_checkpoint_sha256=digest(neural), boundary_checkpoint_sha256=digest(boundary),
         pins={str(path): digest(path) for path in pins}, maximum_new_artifact_allowance_bytes=18_000_000,
         supervisor_log_reservation_bytes=1_000_000, storage_bytes_at_launch=used,
         concurrent_commands=2, workers_per_command=2,
         purpose='Preserve a measured hard-thin printed-wall warning from the existing original neural camera even when no plan passes both views.',
         runtime_change='Opt-in boundary-observed-wall-guard. On nonfit container walls above the specified base, apply the unchanged hard-wall rule to the current learned observation against original nominal geometry; retain any warning across all candidate plans. No fault,seed,onset,severity,truth or outcome enters the policy. No thresholds,physical state,measurement history or model weights change.',
         acceptance='First reproduce both v51 known-defect controls under the new source with the option disabled and identical action layers. Enabled neural and actual-Qwen variants must stop at the observed hard-thin layer101 with zero adaptations, matched camera prefixes and no defective finish. Then preserve all50 v51 rows: every prior success and its action sequence,14fault successes and6clean controls pervariant,zero defective finishes,no new defective case,actual temporal/boundary inference and accepted Qwen calls in the preservation grid. Preserve exact camera prefixes, persistent faults and three-adaptation limit.',
         reserved_fresh_seeds_untouched=[2001, 2002, 2003, 2004],
         limitation='Consumed targeted development/preservation only. Preventing a defective finish is not a recovered successful print. No full-grid or fresh-test superiority and no automatic broad run or promotion.')
assert p['source_sha256'] == runtime_digest(repo)
assert p['simulation_contract_sha256'] == old['simulation_contract_sha256'] == 'fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n')
launch = '#!/bin/bash\nset -uo pipefail\ncd ' + shlex.quote(str(source)) + '\n'
launch += 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch += shlex.join([str(repo / '.venv/bin/python'), '-u', str(root / 'run.py')]) + ' > ' + shlex.quote(str(root / 'run.log')) + ' 2>&1\n'
launch += 'result=$?\nprintf "%s\\n" "$result" > ' + shlex.quote(str(root / 'run.exit')) + '\nexit "$result"\n'
(root / 'launch.sh').write_text(launch)
window = subprocess.check_output(['tmux', 'new-window', '-d', '-t', 'neural-recovery', '-n', 'observed-wall-v62',
                                  '-P', '-F', '#{window_id}', 'bash ' + shlex.quote(str(root / 'launch.sh'))], text=True).strip()
(root / 'launch.json').write_text(json.dumps(dict(tmux_window=window, session='neural-recovery',
                                                protocol_sha256=digest(root / 'protocol.json'), runner_sha256=digest(root / 'run.py')),
                                           indent=2) + '\n')
print(json.dumps(dict(root=str(root), window=window, used=used, source_sha256=p['source_sha256'], maximum_rows=54)))
