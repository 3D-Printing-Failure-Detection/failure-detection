from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source_directory']);sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract


def verify():
    assert runtime_digest(source)==p['source_sha256']
    assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
    for path,expected in p['pins'].items():assert digest(path)==expected,path


def condition(**value):
    (root/'condition.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)


def run(name,command,training=False):
    usage=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)]).split()[0])
    used,own=usage('/mnt/data/ishaan'),usage(root)
    reserved=0
    assert own<p['maximum_new_artifact_allowance_bytes'] and used+max(0,p['maximum_new_artifact_allowance_bytes']-own)+reserved<99_990_000_000
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',
             PYTHONPATH=p['training_pythonpath'] if training else str(source),GIT_DIR='/home/ishaan/failure-detection/.git')
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/(name+'-process.json')).write_text(json.dumps(dict(pid=process.pid,command=command),indent=2)+'\n')
        assert process.wait()==0,name+' failed'
    print('DONE',name,flush=True)


verify();condition(status='building_training_coverage',new_recovery_trials=0)
run('dataset',p['dataset_command'])
dataset=read(root/'dataset/manifest.json')
assert dataset['status']=='complete' and dataset['expected_groups']==64
assert dataset['original_dataset_unchanged'] and dataset['training_only']
counts=sorted((row['critical_positive_pixels'] for row in dataset['groups']),reverse=True)
coverage=dict(critical_positive_pixels=sum(counts),groups_with_at_least_16_critical_pixels=sum(c>=16 for c in counts),
    top_two_group_fraction=sum(counts[:2])/max(1,sum(counts)),
    groups_with_true_open_frames=sum(g['true_open_frames']>0 for g in dataset['groups']),rows=dataset['rows'])
(root/'coverage-audit.json').write_text(json.dumps(coverage,indent=2)+'\n')
assert coverage['groups_with_at_least_16_critical_pixels']>=8 and coverage['top_two_group_fraction']<=.5
assert coverage['groups_with_true_open_frames']>=8
condition(status='training_matched_control',new_recovery_trials=0)
run('control',p['training_commands']['control'],True)
assert digest(root/'control/boundary.npz')==p['reference_control_sha256'],'original control checkpoint did not reproduce'
condition(status='training_coverage_candidate',new_recovery_trials=0,control_reproduced_exactly=True)
run('candidate',p['training_commands']['candidate'],True)
control,candidate=read(root/'control/training.json'),read(root/'candidate/training.json')
for d in [control,candidate]:
    assert d['status']=='complete' and d['validation_rows']==2040 and d['samples_per_epoch']==3720
    assert len(d['validation_groups'])==36 and not set(d['training_groups'])&set(d['validation_groups'])
    assert not d['include_previous_expected_mask'] and d['critical_material_training'] is None
assert control['train_rows']==3720 and len(control['training_groups'])==66 and control['extra_training'] is None
assert candidate['train_rows']==3720+dataset['rows'] and len(candidate['training_groups'])==130
assert set(control['training_groups'])<=set(candidate['training_groups'])
assert candidate['validation_groups']==control['validation_groups']
assert candidate['initialization_sha256']==control['initialization_sha256']
assert candidate['extra_training']['extra_sampling_mass']==.5
assert candidate['extra_training']['original_validation_unchanged'] and candidate['extra_training']['training_only']
assert candidate['extra_training']['sha256']==dataset['data_sha256']
condition(status='complete_print_camera_diagnostics',new_recovery_trials=0,control_reproduced_exactly=True)
diagnostics=[]
for job in p['diagnostics']:
    run(job['name'],job['command'])
    final=read(Path(job['out'])/'final.json');manifest=read(Path(job['out'])/'manifest.json')
    assert manifest['status']=='complete' and manifest['frames']==160
    assert final['truth']['ok'],'retained ignore success did not reproduce'
    diagnostics.append(dict(case=job['case'],passes=final['measured']['coverage:classical-base']['ok'],
        final=final,manifest_sha256=digest(Path(job['out'])/'manifest.json'),final_sha256=digest(Path(job['out'])/'final.json')))
    (root/'diagnostic-results.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
if not all(d['passes'] for d in diagnostics):
    verify();condition(status='rejected',reason='at least one complete-print camera diagnostic still fails',
        diagnostic_frames=640,new_recovery_trials=0,control_reproduced_exactly=True,
        candidate_checkpoint_sha256=candidate['checkpoint_sha256'],diagnostics=diagnostics)
    sys.exit(0)
condition(status='broad_camera_validation',new_recovery_trials=0,diagnostic_frames=640)
run('geometry',p['geometry_command'])
reference=[json.loads(l) for l in Path(p['reference_geometry_frames']).read_text().splitlines()]
reference={(r['group'],r['profile'],r['layer']):r for r in reference}
rows=[json.loads(l) for l in (root/'geometry-validation/frames.jsonl').read_text().splitlines()]
assert len(rows)==2040
fatal=lambda v:any(v.get(k,False) for k in ['open_wall','hard_thin','base_problem','connection_problem'])
counts={};lost=[]
for model in ['classical','original','control','coverage']:
    c=Counter()
    for row in rows:
        ref=reference[(row['group'],row['profile'],row['layer'])]
        assert row['truth']==ref['truth'] and row['measured']['classical']==ref['measured']['classical']
        assert row['measured']['original']==ref['measured']['original']
        measured=(row['measured']['classical'] if model in ['control','coverage'] and 'base_problem' in row['truth'] else row['measured'][model])
        truth,predicted=fatal(row['truth']),fatal(measured)
        c[('true_positive' if truth else 'false_positive') if predicted else ('false_negative' if truth else 'true_negative')]+=1
        if model=='coverage' and truth and fatal(row['measured']['original']) and not predicted:
            lost.append([row['group'],row['profile'],row['layer']])
    counts[model]=dict(c)
passes=not lost and counts['coverage']['false_negative']<=counts['original']['false_negative']
verify();condition(status='camera_gates_passed' if passes else 'rejected',new_recovery_trials=0,
    diagnostic_frames=640,validation_frames=2040,control_reproduced_exactly=True,all_prior_geometry_controls_reproduced=True,
    candidate_checkpoint_sha256=candidate['checkpoint_sha256'],functional_detection_counts=counts,
    lost_original_true_detections=lost,limitation=p['limitation'])
