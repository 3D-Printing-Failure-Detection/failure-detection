from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys,time
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
    other=Path(p['coverage_root'])
    reserved=max(0,28_000_000-usage(other)) if not (other/'run.exit').exists() else 0
    assert own<p['maximum_new_artifact_allowance_bytes'] and used+max(0,p['maximum_new_artifact_allowance_bytes']-own)+reserved<99_990_000_000
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',
             PYTHONPATH=p['training_pythonpath'] if training else str(source),GIT_DIR='/home/ishaan/failure-detection/.git')
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/(name+'-process.json')).write_text(json.dumps(dict(pid=process.pid,command=command),indent=2)+'\n')
        assert process.wait()==0,name+' failed'
    print('DONE',name,flush=True)


verify();condition(status='extending_critical_material_labels',new_recovery_trials=0)
run('dataset',p['dataset_command'])
labels=read(root/'dataset/manifest.json');dataset=read(Path(p['coverage_root'])/'dataset/manifest.json')
assert labels['status']=='complete' and labels['rows']==7104 and labels['verified_augmented_rows']==1344
assert labels['original_labels_unchanged'] and labels['original_validation_unchanged']
assert labels['critical_training_pixels']==31738 and labels['critical_validation_pixels']==1147
assert labels['extra_data_sha256']==dataset['data_sha256']
condition(status='training_matched_control',new_recovery_trials=0)
run('control',p['training_commands']['control'],True)
assert digest(root/'control/boundary.npz')==p['reference_control_sha256'],'original control checkpoint did not reproduce'
condition(status='training_combined_candidate',new_recovery_trials=0,control_reproduced_exactly=True)
run('candidate',p['training_commands']['candidate'],True)
control,candidate=read(root/'control/training.json'),read(root/'candidate/training.json')
for d in [control,candidate]:
    assert d['status']=='complete' and d['validation_rows']==2040 and d['samples_per_epoch']==3720
    assert d['train_rows']==5064 and len(d['training_groups'])==130 and len(d['validation_groups'])==36
    assert not set(d['training_groups'])&set(d['validation_groups']) and not d['include_previous_expected_mask']
    assert d['extra_training']['extra_sampling_mass']==.5 and d['extra_training']['original_validation_unchanged']
    assert d['extra_training']['training_only'] and d['extra_training']['sha256']==dataset['data_sha256']
assert control['critical_material_training'] is None
assert candidate['critical_material_training']['desired_critical_loss_mass']==.5
assert candidate['critical_material_training']['critical_training_pixels']==31738
assert candidate['training_groups']==control['training_groups'] and candidate['validation_groups']==control['validation_groups']
assert candidate['initialization_sha256']==control['initialization_sha256']
condition(status='complete_print_camera_diagnostics',new_recovery_trials=0,control_reproduced_exactly=True)
diagnostics=[]
for job in p['diagnostics']:
    run(job['name'],job['command'])
    final=read(Path(job['out'])/'final.json');manifest=read(Path(job['out'])/'manifest.json')
    assert manifest['status']=='complete' and manifest['frames']==160
    assert final['truth']['ok'],'retained ignore success did not reproduce'
    reference_path=Path(p['coverage_root'])/job['name']/'final.json'
    while not reference_path.exists() or not (reference_path.parent/'manifest.json').exists():
        assert not (Path(p['coverage_root'])/'run.exit').exists(),'reference terminated without final diagnostic'
        assert str(Path(p['coverage_root'])/'run.py').encode() in Path(f"/proc/{p['coverage_runner_pid']}/cmdline").read_bytes().split(b'\0')
        time.sleep(5)
    assert read(reference_path.parent/'manifest.json')['status']=='complete'
    reference=read(reference_path)
    assert final['truth']==reference['truth'] and final['measured']['classical']==reference['measured']['classical']
    assert final['measured']['control']==reference['measured']['coverage']
    assert final['measured']['control:classical-base']==reference['measured']['coverage:classical-base']
    (root/(job['name']+'-control-reproduction.json')).write_text(json.dumps(dict(status='verified',
        reference=str(reference_path),reference_sha256=digest(reference_path),
        current_final_sha256=digest(Path(job['out'])/'final.json')),indent=2)+'\n')
    diagnostics.append(dict(case=job['case'],passes=final['measured']['combined:classical-base']['ok'],
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
for model in ['classical','original','control','combined']:
    c=Counter()
    for row in rows:
        ref=reference[(row['group'],row['profile'],row['layer'])]
        assert row['truth']==ref['truth'] and row['measured']['classical']==ref['measured']['classical']
        assert row['measured']['original']==ref['measured']['original']
        measured=(row['measured']['classical'] if model in ['control','combined'] and 'base_problem' in row['truth'] else row['measured'][model])
        truth,predicted=fatal(row['truth']),fatal(measured)
        c[('true_positive' if truth else 'false_positive') if predicted else ('false_negative' if truth else 'true_negative')]+=1
        if model=='combined' and truth and fatal(row['measured']['original']) and not predicted:
            lost.append([row['group'],row['profile'],row['layer']])
    counts[model]=dict(c)
passes=not lost and counts['combined']['false_negative']<=counts['original']['false_negative']
verify();condition(status='camera_gates_passed' if passes else 'rejected',new_recovery_trials=0,
    diagnostic_frames=640,validation_frames=2040,control_reproduced_exactly=True,all_prior_geometry_controls_reproduced=True,
    candidate_checkpoint_sha256=candidate['checkpoint_sha256'],functional_detection_counts=counts,
    lost_original_true_detections=lost,limitation=p['limitation'])
