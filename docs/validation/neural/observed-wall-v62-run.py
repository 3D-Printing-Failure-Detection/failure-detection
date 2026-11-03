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
p = read(root / 'protocol.json')
source = Path(p['source'])
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract
from tools.result_storage import load_run_bytes
from recovery.eval.camera_trace import validate_camera_prefixes


def verify():
    assert runtime_digest(source) == p['source_sha256']
    assert simulation_contract(source)['sha256'] == p['simulation_contract_sha256']
    for path, expected in p['pins'].items():
        assert digest(path) == expected, path


def save(**value):
    (root / 'condition.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def execute(job):
    size = lambda path: int(subprocess.check_output(['du', '-sx', '-B1', str(path)]).split()[0])
    used, own = size('/mnt/data/ishaan'), size(root)
    assert own < p['maximum_new_artifact_allowance_bytes']
    assert used + max(0, p['maximum_new_artifact_allowance_bytes'] - own) + p['supervisor_log_reservation_bytes'] < 99_990_000_000
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='egl',
               PYTHONPATH=str(source), GIT_DIR='/home/ishaan/failure-detection/.git')
    print('START', job['name'], flush=True)
    with (root / (job['name'] + '.log')).open('w') as log:
        process = subprocess.Popen(job['command'], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        (root / (job['name'] + '-process.json')).write_text(json.dumps(dict(pid=process.pid, command=job['command']), indent=2) + '\n')
        code = process.wait()
        (root / (job['name'] + '.exit')).write_text(str(code) + '\n')
        assert code == 0, job['name']
    directory = root / job['name'] / 'results'
    with (root / (job['name'] + '-audit.log')).open('w') as log:
        subprocess.run([sys.executable, str(source / 'tools/audit_neural_experiment.py'), str(directory),
                        '--allow-unreached-agent', '--require-camera-traces', '--out', str(directory.parent / 'audit')],
                       cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest, audit = read(directory / 'manifest.json'), read(directory.parent / 'audit/audit.json')
    raw = load_run_bytes(directory)
    assert audit['status'] == 'passed' and audit['runs_sha256'] == hashlib.sha256(raw).hexdigest()
    assert manifest['source_sha256'] == manifest['final_source_sha256'] == p['source_sha256']
    assert manifest['completed_runs'] == job['expected_rows']
    assert manifest['boundary_checkpoint_sha256'] == p['boundary_checkpoint_sha256']
    assert manifest['neural_checkpoint_sha256'] == p['neural_checkpoint_sha256']
    assert manifest['arguments']['boundary_observed_wall_guard'] == job['enabled']
    assert manifest['arguments']['boundary_trend_continuation_fallback']
    assert not manifest['arguments']['contour_trend_shrinkage'] and not manifest['arguments']['boundary_classical_base']
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == job['expected_rows']
    assert all(row['adaptations'] <= 3 and not row['clear_failure_on_adapt'] for row in rows)
    print('AUDITED', job['name'], flush=True)
    return rows


def batches(jobs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(execute, jobs))


key = lambda row: tuple(row[k] for k in ['profile', 'part', 'kind', 'seed', 'method'])
fields = ['profile', 'part', 'kind', 'seed', 'method', 'in_place', 'defective_finished', 'adaptations',
          'layers_printed', 'boundary_queries', 'neural_queries', 'agent_requests', 'wall_guard_checks', 'wall_guard_findings']
compact = lambda row: {k: row[k] for k in fields}


def main():
    verify()
    reference = {}
    for directory in map(Path, p['reference_directories']):
        raw, audit = load_run_bytes(directory), read(directory.parent / 'audit/audit.json')
        assert audit['status'] == 'passed' and audit['runs_sha256'] == hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            row = json.loads(line)
            assert key(row) not in reference
            reference[key(row)] = row
    assert set(reference) == {tuple(case) + (method,) for case in p['preservation_cases'] for method in p['methods']}
    assert len(reference) == 50

    save(status='paired_known_defect_recovery', new_rows=0)
    control_rows, candidate_rows = batches(p['target_jobs'])
    expected = {tuple(p['target_case']) + (method,) for method in p['methods']}
    assert {key(r) for r in control_rows} == {key(r) for r in candidate_rows} == expected
    for row in control_rows:
        old = reference[key(row)]
        for field in ['in_place', 'defective_finished', 'layers_printed', 'adaptations', 'k0', 'severity', 'n_layers', 'clear_failure_on_adapt']:
            assert row[field] == old[field], ('control mismatch', field)
        assert [(e['layer'], e['action']) for e in row['actions']] == [(e['layer'], e['action']) for e in old['actions']]
        assert row['defective_finished'] and not row['wall_guard_findings']
    for row in candidate_rows:
        old = reference[key(row)]
        for field in ['k0', 'severity', 'n_layers', 'clear_failure_on_adapt']:
            assert row[field] == old[field]
    target_passes = all(row['outcome'] == 'scrapped' and not row['defective_finished']
                        and row['layers_printed'] == p['target_hard_wall_layer'] + 1
                        and row['adaptations'] == 0
                        and len(row['wall_guard_findings']) == 1
                        and row['wall_guard_findings'][0]['layer'] == p['target_hard_wall_layer']
                        for row in candidate_rows)
    combined = control_rows + candidate_rows + [reference[k] for k in expected]
    paired = dict(passes=target_passes, control=[compact(r) for r in control_rows],
                  candidate=[compact(r) for r in candidate_rows],
                  camera_consistency=validate_camera_prefixes(combined, required=True),
                  limitation='Stopping a pre-existing defective finish is not a successful recovered print.')
    (root / 'paired-comparison.json').write_text(json.dumps(paired, indent=2) + '\n')
    if not target_passes:
        verify()
        save(status='rejected', reason='wall guard does not stop the paired known defect at the observed hard-thin layer', new_rows=4)
        return

    save(status='preservation_recovery', new_rows=4, paired_gate_passed=True)
    done = batches(p['preservation_jobs'])
    rows = [row for items in done for row in items]
    assert len(rows) == 50 and {key(r) for r in rows} == set(reference)
    for row in rows:
        for field in ['k0', 'severity', 'n_layers', 'clear_failure_on_adapt']:
            assert row[field] == reference[key(row)][field]
    methods = {}
    for method in p['methods']:
        selected = [r for r in rows if r['method'] == method]
        faults, clean = [r for r in selected if r['kind'] != 'clean'], [r for r in selected if r['kind'] == 'clean']
        lost = [list(key(r)) for r in selected if reference[key(r)]['in_place'] and not r['in_place']]
        added = [list(key(r)) for r in selected if r['defective_finished'] and not reference[key(r)]['defective_finished']]
        action_changes = [list(key(r)) for r in selected if reference[key(r)]['in_place'] and
                          [(e['layer'], e['action']) for e in r['actions']] !=
                          [(e['layer'], e['action']) for e in reference[key(r)]['actions']]]
        result = dict(fault_cases=len(faults), fault_successes=sum(r['in_place'] for r in faults),
                      defective_finishes=sum(r['defective_finished'] for r in faults), clean_cases=len(clean),
                      clean_successes=sum(r['in_place'] for r in clean), lost_reference_successes=lost,
                      new_defects=added, successful_reference_action_changes=action_changes,
                      neural_queries=sum(r['neural_queries'] for r in selected), boundary_queries=sum(r['boundary_queries'] for r in selected),
                      agent_requests=sum(r['agent_requests'] for r in selected),
                      wall_guard_findings=[dict(case=list(key(r)), findings=r['wall_guard_findings']) for r in selected if r['wall_guard_findings']])
        result['passes'] = bool(not lost and not added and not action_changes and result['fault_successes'] >= 14
                                and result['defective_finishes'] == 0 and result['clean_successes'] == len(clean) == 6
                                and result['neural_queries'] > 0 and result['boundary_queries'] > 0
                                and (not method.endswith('_agent') or result['agent_requests'] > 0))
        methods[method] = result
    report = dict(status='passed' if all(r['passes'] for r in methods.values()) else 'rejected', new_rows=54,
                  targeted=paired, preservation=methods, source_sha256=p['source_sha256'],
                  camera_consistency=validate_camera_prefixes(list(reference.values()) + control_rows + candidate_rows + rows, required=True),
                  limitation=p['limitation'])
    (root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    verify()
    save(status=report['status'], new_rows=54, comparison_sha256=digest(root / 'comparison.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(status='failed', error=repr(exc))
        raise
