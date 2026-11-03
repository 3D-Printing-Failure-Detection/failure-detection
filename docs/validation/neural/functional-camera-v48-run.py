from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
p = json.loads((root/'protocol.json').read_text())
source = Path(p['source_directory'])
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract

read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    assert runtime_digest(source) == p['source_sha256']
    assert simulation_contract(source)['sha256'] == p['simulation_contract_sha256']
    for path, expected in p['pins'].items():
        assert digest(path) == expected, path


verify()
frames = [json.loads(line) for line in Path(p['geometry_frames']).read_text().splitlines()]
assert len(frames) == 2040
flags = ['open_wall', 'hard_thin', 'base_problem', 'connection_problem']
fatal = lambda value: any(value.get(k, False) for k in flags)
scores = {}
lost = []
extra_gap_misses = []
for name in ['classical', 'original', 'control', 'nonlinear']:
    counts = Counter()
    for row in frames:
        truth = fatal(row['truth'])
        measured = dict(row['measured'][name])
        if name in ['control', 'nonlinear'] and 'base_problem' in row['truth']:
            measured = row['measured']['classical']
        predicted = fatal(measured)
        counts[('true_positive' if truth else 'false_positive') if predicted
               else ('false_negative' if truth else 'true_negative')] += 1
        if name == 'nonlinear' and truth and fatal(row['measured']['original']) and not predicted:
            lost.append([row['group'], row['profile'], row['layer']])
    scores[name] = dict(counts)
for row in frames:
    if (row['truth'].get('open_wall') and row['measured']['original'].get('open_wall')
            and not row['measured']['nonlinear'].get('open_wall')):
        extra_gap_misses.append(dict(group=row['group'], profile=row['profile'], layer=row['layer'],
            refined_hard_thin=row['measured']['nonlinear'].get('hard_thin', False)))
geometry_gate = (not lost
    and scores['nonlinear']['false_negative'] <= scores['original']['false_negative']
    and all(row['refined_hard_thin'] for row in extra_gap_misses))
geometry = dict(status='complete', frames=2040, flags=flags, counts=scores,
    original_fatal_detections_lost=lost, extra_gap_label_misses=extra_gap_misses,
    passes=geometry_gate,
    limitation='Retrospective paired frame analysis. These are sufficient per-layer causes of functional rejection; soft-thin budgets and inter-layer support require complete-print checks. True base and connection faults are absent from these selected frames. This does not reverse the failed per-feature v35 gate or prove recovery safety.')
(root/'geometry-functional-comparison.json').write_text(json.dumps(geometry, indent=2)+'\n')
print(json.dumps(geometry), flush=True)
assert geometry_gate
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
    OMP_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONPATH=str(source),
    GIT_DIR='/home/ishaan/failure-detection/.git')
results = []
for job in p['jobs']:
    print('START', job['name'], flush=True)
    with (root/(job['name']+'.log')).open('w') as log:
        process = subprocess.Popen(job['command'], cwd=source, env=env,
            stdout=log, stderr=subprocess.STDOUT)
        (root/(job['name']+'-process.json')).write_text(json.dumps(dict(pid=process.pid,
            command=job['command']), indent=2)+'\n')
        assert process.wait() == 0, job['name']
    out = Path(job['out'])
    manifest = read(out/'manifest.json')
    verdict = read(out/'final.json')
    assert manifest['status'] == 'complete' and manifest['frames'] == 160
    assert manifest['source_sha256'] == manifest['final_source_sha256'] == p['source_sha256']
    assert verdict['truth']['ok'], 'paired ignore success did not reproduce'
    results.append(dict(case=job['case'], verdict=verdict,
        passes=verdict['measured']['nonlinear:classical-base']['ok'],
        files={str(out/f): digest(out/f) for f in ['manifest.json', 'final.json', 'layers.json']}))
    print('DONE', job['name'], results[-1]['passes'], flush=True)
verify()
known = read(p['known_cup_verdict'])
assert known['truth']['ok'] and known['measured']['nonlinear:classical-base']['ok']
summary = dict(status='passed' if all(r['passes'] for r in results) else 'rejected',
    new_diagnostic_frames=480, new_recovery_trials=0, geometry_gate=geometry,
    known_cup_verdict=known, new_cases=results,
    all_four_consumed_ignore_successes_pass=all(r['passes'] for r in results),
    protocol_sha256=digest(root/'protocol.json'), source_sha256=p['source_sha256'],
    limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(dict(status=summary['status'], cases=[dict(case=r['case'], passes=r['passes']) for r in results])), flush=True)
