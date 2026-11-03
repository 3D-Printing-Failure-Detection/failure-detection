from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys

root=Path('/mnt/data/ishaan/hri-workspace/failure-detection/neural-trend-full-v38')
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source'])
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract,load_configurations,verify_configuration
assert int((root/'run.exit').read_text())==1
assert not (root/'completion.json').exists()
assert runtime_digest(source)==p['source_sha256']
assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
for path,expected in p['pins'].items():assert digest(path)==expected,path
assert digest(root/'configurations.json')==p['configurations_sha256']
assert digest(root/'labels.json')==p['labels_sha256']
labels=read(root/'labels.json');before=read(root/'labels.json')
prior_root=Path(p['prior_root'])
prior=read(prior_root/'comparison.json')
previous=str(prior_root/'full/results')
assert labels[previous]=={}
labels[previous]=dict(p['prior_label_mapping'])
assert set(labels[previous])=={'neural_risk','neural_risk_agent'}
assert labels[previous]=={'neural_risk':'robust_camera_neural','neural_risk_agent':'robust_camera_agent'}
assert {k:v for k,v in labels.items() if k!=previous}=={k:v for k,v in before.items() if k!=previous}
(root/'labels-report-repair.json').write_text(json.dumps(labels,indent=2)+'\n')
configs,_=load_configurations(root/'configurations.json')
input_pins={};used=set()
for value in p['inputs']:
    directory=Path(value);manifest=read(directory/'manifest.json');audit_path=directory.parent/'audit/audit.json';audit=read(audit_path)
    assert manifest['status']=='complete' and audit['status']=='passed'
    assert digest(directory/'runs.jsonl')==audit['runs_sha256']
    assert manifest['completed_runs']==manifest['expected_runs']==audit['rows']
    for method in manifest['arguments']['methods']:
        label=labels.get(str(directory.resolve()),{}).get(method,method)
        verify_configuration(configs['configurations'][label],method,manifest,manifest_sha256=digest(directory/'manifest.json'))
        used.add(label)
    input_pins.update({str(path):digest(path) for path in [directory/'runs.jsonl',directory/'manifest.json',audit_path]})
assert used==set(configs['configurations']) and len(used)==25
repair=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    reason='The original reporting labels omitted the already-preregistered renaming of the two v33 rows, pooling old and new runtimes under the same names. The strict source check rejected that pooling after all 864 new simulations passed audit.',
    original_run_exit=1,original_run_exit_sha256=digest(root/'run.exit'),
    original_protocol_sha256=digest(root/'protocol.json'),
    original_labels_sha256=digest(root/'labels.json'),repaired_labels_sha256=digest(root/'labels-report-repair.json'),
    configuration_sha256=digest(root/'configurations.json'),
    changed_directory=previous,old_mapping=before[previous],new_mapping=labels[previous],
    preflight_all_25_configurations_verified=True,input_pins=input_pins,new_simulations=0,
    simulation_results_manifests_audits_unchanged=True,
    script_sha256=digest(Path(__file__)))
(root/'reporting-repair.json').write_text(json.dumps(repair,indent=2)+'\n')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONPATH=str(source),GIT_DIR='/home/ishaan/failure-detection/.git')
for candidate,name in [('neural_risk_agent','comparison.json'),('neural_risk','neural-only-comparison.json')]:
    command=[sys.executable,str(source/'tools/compare_neural_experiments.py'),*p['inputs'],
        '--configurations',str(root/'configurations.json'),'--labels',str(root/'labels-report-repair.json'),
        '--candidate',candidate,'--baselines',*p['baselines'],'--failure-cases','240','--out',str(root/name)]
    with (root/(name+'.resume.log')).open('w') as log:
        subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    current=read(root/name)
    assert len(current['summary'])==25
    assert sum(split['n'] for result in current['summary'].values() for split in result.values())==7200
    for label,result in prior['summary'].items():
        assert current['summary'][p['prior_label_mapping'].get(label,label)]==result
with (root/'attribution.resume.log').open('w') as log:
    subprocess.run([sys.executable,p['attribution_script'],str(root/'comparison.json'),str(root/'attribution.json')],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for path,expected in input_pins.items():assert digest(path)==expected,path
current=read(root/'comparison.json')
result=dict(status='complete',new_rows=864,rows=7200,all_22_previous_summaries_identical=True,
    comparison_sha256=digest(root/'comparison.json'),summary=current['summary'],comparisons=current['comparisons'],
    **{key:current[key] for key in ['actual_learning','actual_agent','strictly_more_completions_than_every_baseline',
        'no_lost_baseline_successes','no_higher_defective_finish_count','all_candidate_clean_controls_pass']},
    reporting_repair_sha256=digest(root/'reporting-repair.json'),limitation=p['limitation'])
(root/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'report-resume.exit').write_text('0\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['summary','comparisons']},indent=2))
