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
root = base/'neural-continuation-fallback-v51'
oldroot = base/'neural-trend-shrinkage-v37'
old = json.loads((oldroot/'protocol.json').read_text())
size = lambda path: int(subprocess.check_output(['du', '-sx', '-B1', str(path)], text=True).split()[0])
used = size('/mnt/data/ishaan')
assert used+13_000_000 < 99_990_000_000, used
assert (base/'neural-command-history-v50/run.exit').read_text().strip() == '0'
assert (base/'neural-trend-historical-v43/report-resume.exit').read_text().strip() == '0'
supervisor_path = base/'local-supervisor-worker-v1/manifest.json'
supervisor = json.loads(supervisor_path.read_text())
actual_command = Path('/proc/2524739/cmdline').read_bytes().decode().strip('\0').split('\0')
assert actual_command == supervisor['server_command']
with Path(supervisor['reuses_model_file']).open('rb') as f:
    model_digest = hashlib.file_digest(f, 'sha256').hexdigest()
assert model_digest == supervisor['verified_model_sha256']
with urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=10) as response:
    health = json.load(response)
assert health['status'] == 'ok'
root.mkdir()
source = root/'source'
for name in ['recovery', 'parts', 'tools']:
    shutil.copytree(repo/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
sys.path.insert(0, str(source))
from tools.experiment_provenance import file_digest, runtime_digest, simulation_contract

tests = dict(repository=dict(passed=212, skipped=3, seconds=28.10,
    command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -m pytest -q'),
    focused=dict(passed=48, seconds=1.20,
    command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_damped_continuation.py tests/test_camera_fallback.py tests/test_contour_trend.py tests/test_experiment_audit.py'),
    source_sha256=runtime_digest(source), test_sha256={str(path.relative_to(repo)):file_digest(path)
        for path in [repo/'tests/test_damped_continuation.py', repo/'tests/test_camera_fallback.py',
                     repo/'tests/test_contour_trend.py', repo/'tests/test_experiment_audit.py']})
(root/'tests.json').write_text(json.dumps(tests, indent=2)+'\n')
attestation = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=2524739,
    server_command=actual_command, health=health, model_sha256=model_digest,
    manifest_sha256=file_digest(supervisor_path))
(root/'supervisor-runtime.json').write_text(json.dumps(attestation, indent=2)+'\n')
common = old['control_job']['command'][:old['control_job']['command'].index('--methods')]
methods = ['neural_risk', 'neural_risk_agent']
paired_jobs = []
groups = [('shift-recoveries', ['cartesian', 'corexy'], 'shift', 10),
          ('under-recovery', ['cartesian'], 'under_extrusion', 23),
          ('mug-gain', ['bedslinger'], 'collapse', 1001)]
for enabled in (False, True):
    for name, profiles, kind, seed in groups:
        name = ('candidate-' if enabled else 'control-')+name
        command = common + ['--methods', *methods, '--n', '1', '--workers', '2',
            '--profiles', *profiles, '--parts', 'mug', '--kinds', kind, '--seed', str(seed),
            '--out', str(root/name/'results')]
        if enabled:
            command.append('--boundary-trend-continuation-fallback')
        paired_jobs.append(dict(name=name, command=command, expected_rows=2*len(profiles), enabled=enabled))
regression = []
for job in old['commands']:
    command = [arg.replace(str(oldroot), str(root)) for arg in job['command']
               if arg != '--contour-trend-shrinkage']
    command.append('--boundary-trend-continuation-fallback')
    regression.append(dict(name=job['name'], command=command, expected_rows=job['expected_rows'], enabled=True))
paired_references = [str(oldroot/'mug-control/results'),
    str(base/'neural-robust-historical-v34/main-containers/results'),
    str(base/'neural-robust-historical-v34/heldout-containers/results')]
pins = {}
for directory in map(Path, paired_references+old['references']):
    for path in [directory/'runs.jsonl', directory/'manifest.json', directory.parent/'audit/audit.json']:
        pins[str(path)] = file_digest(path)
for path in [source/'tools/audit_neural_experiment.py', source/'tools/experiment_provenance.py',
    source/'tools/result_storage.py', root/'tests.json', root/'supervisor-runtime.json', supervisor_path,
    base/'neural-equivariant-v1/classical/temporal.npz', base/'neural-equivariant-v1/classical/training.json',
    base/'neural-boundary-v1/model/boundary.npz', base/'neural-boundary-v1/model/training.json']:
    pins[str(path)] = file_digest(path)
p = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage='development',
    source=str(source), source_sha256=runtime_digest(source),
    simulation_contract_sha256=simulation_contract(source)['sha256'], methods=methods,
    purpose='Retain original passing recovery plans and test damped dual-camera continuation only after all original plans fail',
    paired_cases=[[profile, 'mug', kind, seed] for _, profiles, kind, seed in groups for profile in profiles],
    new_gain_case=['bedslinger', 'mug', 'collapse', 1001], paired_jobs=paired_jobs,
    paired_references=paired_references, regression_cases=old['cases'][:-1], regression_jobs=regression,
    regression_references=old['references'], pins=pins, expected_paired_rows=16,
    expected_total_rows_if_paired_gate_passes=58, allowed_existing_defective_cases=old['allowed_existing_defective_cases'],
    concurrent_commands=2, workers_per_command=2, maximum_new_artifact_allowance_bytes=12_000_000,
    storage_bytes_at_launch=used, additional_supervisor_log_reservation_bytes=1_000_000,
    change='Original candidate evaluation and selection are unchanged whenever any original plan passes and is executable. Otherwise, for nonfits only, cloned analytical camera histories test the unmodified remaining command under damped trends; both full-spec checks must pass. No damped corrections, prefix changes, extra adaptation budget or new thresholds.',
    acceptance='Reproduce all four original controls including action sequences, outcomes and IoU; both variants must recover all three historical mugs with unchanged actions and zero fallback checks, and finish the extra development mug with actual damping. Then retain all ten prior targeted fault successes and six clean controls, no new defective case, and actual trained fit, boundary and Qwen inference. Strict camera traces are required throughout.',
    limitation='Consumed targeted regression only; passing it does not establish full-grid, historical or untouched superiority. Four known camera failures remain separate. No fresh seeds or automatic broad evaluation.')
assert p['simulation_contract_sha256'] == old['simulation_contract_sha256']
(root/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
shutil.copy2('/tmp/continuation-fallback-v51-run.py', root/'run.py')
shutil.copy2(__file__, root/'prepare.py')
launch = '#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch += 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch += shlex.join([str(repo/'.venv/bin/python'), '-u', str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch += 'result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window = subprocess.check_output(['tmux', 'new-window', '-d', '-t', 'neural-recovery', '-n',
    'continuation-v51', '-P', '-F', '#{window_id}', 'bash '+shlex.quote(str(root/'launch.sh'))], text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window, session='neural-recovery',
    protocol_sha256=file_digest(root/'protocol.json'), runner_sha256=file_digest(root/'run.py')), indent=2)+'\n')
print(json.dumps(dict(root=str(root), source_sha256=p['source_sha256'], tmux_window=window,
                     storage_bytes_at_launch=used, maximum_new_bytes=13_000_000)))
