from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = read(root/'protocol.json')
full, prior, source = Path(p['full_root']), Path(p['prior_root']), Path(p['source'])
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract, load_configurations


def verify():
    assert runtime_digest(source) == p['source_sha256']
    assert simulation_contract(source)['sha256'] == p['simulation_contract_sha256']
    for path, expected in p['pins'].items():
        assert digest(path) == expected, path
    for branch in p['branches'].values():
        for comparison in branch.values():
            for kind in ['configurations', 'labels']:
                assert digest(root/comparison[kind+'_file']) == comparison[kind+'_sha256']


def condition(**value):
    (root/'condition.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


def verify_inputs(report):
    for entry in report['inputs']:
        directory = Path(entry['directory'])
        for path, field in [(directory/'manifest.json', 'manifest_sha256'), (directory/'runs.jsonl', 'runs_sha256'),
                            (directory.parent/'audit/audit.json', 'audit_sha256')]:
            assert digest(path) == entry[field], str(path)


verify()
condition(status='waiting_for_full_trend_comparison', new_rows=0)
missing = 0
while not (full/'run.exit').exists():
    try:
        command = Path(f"/proc/{p['full_runner_pid']}/cmdline").read_bytes().split(b'\0')
        assert str(full/'run.py').encode() in command, 'full comparison process identity changed'
        missing = 0
    except FileNotFoundError:
        missing += 1
        assert missing < 3, 'full comparison disappeared without terminal status'
    time.sleep(10)
if int((full/'run.exit').read_text()) != 0 or not (full/'completion.json').exists():
    condition(status='skipped', new_rows=0, reason='full trend comparison did not complete')
    sys.exit(0)
completion = read(full/'completion.json')
assert completion['status'] == 'complete' and completion['rows'] == 7200
assert completion['all_22_previous_summaries_identical']
assert completion['comparison_sha256'] == digest(full/'comparison.json')
full_pins = {}
for name in ['comparison.json', 'neural-only-comparison.json']:
    report = read(full/name)
    verify_inputs(report)
    safe = (report['candidate_has_best_completion_count'] and report['no_higher_defective_finish_count']
            and report['all_candidate_clean_controls_pass'] and report['actual_learning']
            and (not report['candidate'].endswith('_agent') or report['actual_agent'])
            and all(not row['lost'] for row in report['comparisons'] if row['baseline'] != 'ignore'))
    full_pins[name] = digest(full/name)
    if not safe:
        condition(status='skipped', new_rows=0, reason='full trend comparison fails historical prerequisite',
                  comparison=name, comparison_sha256=digest(full/name))
        sys.exit(0)
assert int((prior/'run.exit').read_text()) == 0, 'prior historical orchestration must be repaired first'
prior_terminal = read(prior/'condition.json')
prior_reports = {}
if (prior/'completion.json').exists():
    prior_completion = read(prior/'completion.json')
    assert prior_completion['new_rows'] == 240
    assert prior_completion['status'] in ('passed', 'rejected')
    for name in ['comparison.json', 'fits-comparison.json']:
        report = read(prior/name)
        assert digest(prior/name) == prior_completion['reports'][name]['sha256']
        verify_inputs(report)
        prior_reports[name] = report
    branch = 'with-prior'
else:
    assert prior_terminal['status'] == 'skipped' and prior_terminal['new_rows'] == 0
    assert not list(prior.glob('*/results/runs.jsonl')), 'partial historical results cannot be omitted'
    branch = 'without-prior'
config = p['branches'][branch]
for comparison in config.values():
    load_configurations(root/comparison['configurations_file'])
verification = dict(branch=branch, full_comparison_pins=full_pins, prior_terminal=prior_terminal,
                    prior_report_pins={name: digest(prior/name) for name in prior_reports})
(root/'prerequisite-verification.json').write_text(json.dumps(verification, indent=2)+'\n')
verify()
waiting = False
while True:
    used = int(subprocess.check_output(['du', '-sx', '-B1', '/mnt/data/ishaan'], text=True).split()[0])
    if used+p['maximum_new_artifact_allowance_bytes'] < 99_990_000_000:
        break
    if not waiting:
        condition(status='waiting_for_storage', new_rows=0, storage_bytes=used,
                  required_allowance=p['maximum_new_artifact_allowance_bytes'])
        waiting = True
    time.sleep(30)
condition(status='running', new_rows=0, storage_bytes_at_launch=used, **verification)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MUJOCO_GL='egl', PYTHONPATH=str(source), GIT_DIR='/home/ishaan/failure-detection/.git')


def execute(job):
    print('START', job['name'], flush=True)
    with (root/(job['name']+'.log')).open('w') as log:
        subprocess.run(job['command'], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (root/(job['name']+'-audit.log')).open('w') as log:
        subprocess.run(job['audit_command'], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    directory = root/job['name']/'results'
    assert read(directory.parent/'audit/audit.json')['rows'] == job['expected_rows']
    for row in map(json.loads, (directory/'runs.jsonl').read_text().splitlines()):
        if job['analytical']:
            assert row['neural_queries'] == row['boundary_queries'] == row['agent_requests'] == 0
            assert row['trend_shrinkage_queries'] > 0
        elif row['has_fit_constraints']:
            assert row['trend_shrinkage_queries'] == 0
        else:
            assert row['trend_shrinkage_queries'] > 0
    print('AUDITED', job['name'], flush=True)


with ThreadPoolExecutor(p['concurrent_commands']) as pool:
    list(pool.map(execute, p['commands']))
assert sum(read(root/job['name']/'audit/audit.json')['rows'] for job in p['commands']) == p['expected_new_rows']
summaries = {}
for group, comparison in config.items():
    prefix = '' if group == 'full' else 'fits-'
    for candidate, suffix in [('neural_risk_agent', 'comparison.json'), ('neural_risk', 'neural-only-comparison.json')]:
        name = prefix+suffix
        command = [sys.executable, str(source/'tools/compare_neural_experiments.py'), *comparison['inputs'],
            '--configurations', str(root/comparison['configurations_file']),
            '--labels', str(root/comparison['labels_file']), '--candidate', candidate,
            '--baselines', *comparison['baselines'], '--failure-cases', str(comparison['cases']), '--out', str(root/name)]
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(root/name)
        assert len(report['summary']) == comparison['configurations']
        assert sum(split['n'] for result in report['summary'].values() for split in result.values()) == comparison['expected_rows']
        if prior_reports:
            reference = prior_reports[prefix+'comparison.json']['summary']
            for label, result in reference.items():
                assert report['summary'][p['prior_label_mapping'].get(label, label)] == result
        elif prefix:
            reference = read(p['fits_prior_comparison'])['summary']
            for label, result in reference.items():
                assert report['summary'][p['fits_prior_label_mapping'].get(label, label)] == result
        else:
            for label, splits in p['reference_summary'].items():
                for split, counts in splits.items():
                    for field, value in counts.items():
                        assert report['summary'][label][split][field] == value
        passes = (report['candidate_has_best_completion_count'] and report['no_lost_baseline_successes']
                  and report['no_higher_defective_finish_count'] and report['all_candidate_clean_controls_pass']
                  and report['actual_learning'] and (not candidate.endswith('_agent') or report['actual_agent']))
        summaries[name] = dict(sha256=digest(root/name), passes=bool(passes),
            **{key: report[key] for key in ['summary', 'comparisons', 'actual_learning', 'actual_agent',
               'candidate_has_best_completion_count', 'no_lost_baseline_successes',
               'no_higher_defective_finish_count', 'all_candidate_clean_controls_pass']})
with (root/'attribution.log').open('w') as log:
    subprocess.run([sys.executable, p['attribution_script'], str(root/'comparison.json'), str(root/'attribution.json')],
                   cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
verify()
result = dict(status='passed' if all(report['passes'] for report in summaries.values()) else 'rejected',
    new_rows=360, branch=branch, all_previous_summaries_preserved=True,
    full_comparison_rows=config['full']['expected_rows'], fits_comparison_rows=config['fits-']['expected_rows'],
    reports=summaries, limitation=p['limitation'])
(root/'completion.json').write_text(json.dumps(result, indent=2)+'\n')
condition(status='evaluation_audit_and_comparison_complete', new_rows=360, regression_passes=result['status']=='passed')
