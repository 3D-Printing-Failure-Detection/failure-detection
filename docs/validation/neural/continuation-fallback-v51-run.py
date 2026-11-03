from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = read(root/'protocol.json')
source = Path(p['source'])
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract
from recovery.eval.camera_trace import validate_camera_prefixes

assert runtime_digest(source) == p['source_sha256']
assert simulation_contract(source)['sha256'] == p['simulation_contract_sha256']
for path, expected in p['pins'].items():
    assert digest(path) == expected, path
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
    OMP_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONPATH=str(source),
    GIT_DIR='/home/ishaan/failure-detection/.git')
key = lambda row: tuple(row[k] for k in ('profile', 'part', 'kind', 'seed', 'method'))
action = lambda row: [(a['layer'], a['action']) for a in row['actions']]
condition = dict(status='running_paired_regression', new_rows=0, limitation=p['limitation'])
inputs = []


def save():
    (root/'condition.json').write_text(json.dumps(condition, indent=2)+'\n')
    print(json.dumps(condition), flush=True)


def budget():
    size = lambda path: int(subprocess.check_output(['du', '-sx', '-B1', str(path)], text=True).split()[0])
    used, own = size('/mnt/data/ishaan'), size(root)
    remaining = max(0, p['maximum_new_artifact_allowance_bytes']-own)
    assert own < p['maximum_new_artifact_allowance_bytes'] and used+remaining+1_000_000 < 99_990_000_000


def execute(job):
    budget()
    print('START', job['name'], flush=True)
    with (root/(job['name']+'.log')).open('w') as log:
        subprocess.run(job['command'], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    directory = root/job['name']/'results'
    with (root/(job['name']+'-audit.log')).open('w') as log:
        subprocess.run([sys.executable, str(source/'tools/audit_neural_experiment.py'), str(directory),
            '--allow-unreached-agent', '--require-camera-traces', '--out', str(directory.parent/'audit')],
            cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit, manifest = read(directory.parent/'audit/audit.json'), read(directory/'manifest.json')
    assert audit['status'] == 'passed' and audit['runs_sha256'] == digest(directory/'runs.jsonl')
    assert manifest['source_sha256'] == manifest['final_source_sha256'] == p['source_sha256']
    assert manifest['completed_runs'] == job['expected_rows']
    assert manifest['arguments']['boundary_trend_continuation_fallback'] == job['enabled']
    assert not manifest['arguments']['contour_trend_shrinkage']
    rows = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
    assert all(row['trend_shrinkage_queries'] == 0 for row in rows)
    assert all(row['damped_continuation_checks'] == 0 for row in rows if row['has_fit_constraints'])
    inputs.append(dict(directory=str(directory), manifest_sha256=digest(directory/'manifest.json'),
        runs_sha256=digest(directory/'runs.jsonl'), audit_sha256=digest(directory.parent/'audit/audit.json')))
    print('AUDITED', job['name'], flush=True)
    return rows


def batches(jobs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(execute, jobs))


def reference_rows(directories):
    result = {}
    for directory in map(Path, directories):
        audit = read(directory.parent/'audit/audit.json')
        assert audit['status'] == 'passed' and audit['runs_sha256'] == digest(directory/'runs.jsonl')
        for line in (directory/'runs.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert key(row) not in result
            result[key(row)] = row
    return result


save()
phase = batches(p['paired_jobs'])
control = {key(row): row for job, batch in zip(p['paired_jobs'], phase) if not job['enabled'] for row in batch}
candidate = {key(row): row for job, batch in zip(p['paired_jobs'], phase) if job['enabled'] for row in batch}
expected = {tuple(case)+(method,) for case in p['paired_cases'] for method in p['methods']}
assert set(control) == set(candidate) == expected and len(expected) == 8
reference = reference_rows(p['paired_references'])
for identity, row in control.items():
    old = reference[identity]
    for field in ['in_place', 'defective_finished', 'adaptations', 'layers_printed', 'k0', 'severity', 'measurement_iou']:
        assert row[field] == old[field], ('control differs', identity, field, row[field], old[field])
    assert action(row) == action(old), ('control actions differ', identity)
    assert row['damped_continuation_checks'] == row['damped_continuation_queries'] == 0
for identity, row in candidate.items():
    for field in ['k0', 'severity', 'n_layers', 'clear_failure_on_adapt']:
        assert row[field] == control[identity][field]
    if list(identity[:4]) != p['new_gain_case']:
        assert row['damped_continuation_checks'] == 0
        assert action(row) == action(control[identity]), ('original recovery displaced', identity)
camera = validate_camera_prefixes(list(control.values())+list(candidate.values()), required=True)
passes = all(row['in_place'] and not row['defective_finished'] for row in candidate.values())
passes &= all(row['damped_continuation_queries'] > 0 for identity, row in candidate.items()
              if list(identity[:4]) == p['new_gain_case'])
passes &= any(row['agent_requests'] > 0 for row in candidate.values())
fields = ['profile', 'part', 'kind', 'seed', 'method', 'in_place', 'defective_finished', 'adaptations',
          'layers_printed', 'damped_continuation_checks', 'damped_continuation_queries', 'agent_requests']
paired = dict(passes=bool(passes), control_reproduced=True, camera_consistency=camera,
    control=[{k: row[k] for k in fields} for row in control.values()],
    candidate=[{k: row[k] for k in fields} for row in candidate.values()])
(root/'paired-comparison.json').write_text(json.dumps(paired, indent=2)+'\n')
condition.update(new_rows=16, paired_gate_passes=bool(passes))
if not passes:
    condition.update(status='rejected', regression_rows=0)
    save()
    sys.exit(0)
condition.update(status='running_preservation_regression')
save()
regression = [row for batch in batches(p['regression_jobs']) for row in batch]
reference = reference_rows(p['regression_references'])
expected = {tuple(case)+(method,) for case in p['regression_cases'] for method in p['methods']}
assert {key(row) for row in regression} == set(reference) == expected and len(regression) == 42
for row in regression:
    for field in ['k0', 'severity', 'n_layers', 'clear_failure_on_adapt']:
        assert row[field] == reference[key(row)][field]
summary = {}
for method in p['methods']:
    rows = [row for row in regression if row['method'] == method]
    faults = [row for row in rows if row['kind'] != 'clean']
    clean = [row for row in rows if row['kind'] == 'clean']
    lost = [list(key(row)[:-1]) for row in rows if reference[key(row)]['in_place'] and not row['in_place']]
    new_defects = [list(key(row)[:-1]) for row in faults if row['defective_finished']
                   and list(key(row)[:-1]) not in p['allowed_existing_defective_cases']]
    result = dict(fault_cases=len(faults), fault_successes=sum(row['in_place'] for row in faults),
        defective_finishes=sum(row['defective_finished'] for row in faults), clean_cases=len(clean),
        clean_successes=sum(row['in_place'] for row in clean), lost_reference_successes=lost,
        prohibited_defective_finishes=new_defects,
        neural_queries=sum(row['neural_queries'] for row in rows),
        boundary_queries=sum(row['boundary_queries'] for row in rows),
        agent_requests=sum(row['agent_requests'] for row in rows))
    result['passes'] = bool(not lost and not new_defects and result['fault_successes'] >= 10
        and result['defective_finishes'] <= 1 and result['clean_successes'] == len(clean) == 6
        and result['neural_queries'] > 0 and result['boundary_queries'] > 0
        and (not method.endswith('_agent') or result['agent_requests'] > 0))
    summary[method] = result
    passes &= result['passes']
all_rows = list(control.values())+list(candidate.values())+regression
report = dict(status='passed' if passes else 'rejected', new_rows=len(all_rows), summary=summary,
    source_sha256=p['source_sha256'], camera_consistency=validate_camera_prefixes(all_rows, required=True),
    inputs=sorted(inputs, key=lambda x:x['directory']), paired_comparison_sha256=digest(root/'paired-comparison.json'),
    limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
assert runtime_digest(source) == p['source_sha256']
for path, expected_hash in p['pins'].items():
    assert digest(path) == expected_hash
condition.update(status=report['status'], new_rows=len(all_rows), regression_rows=len(regression),
                 comparison_sha256=digest(root/'comparison.json'))
save()
