from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys

root=Path(__file__).resolve().parent
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source']);sys.path.insert(0,str(source))
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
    assert own<p['maximum_new_artifact_allowance_bytes']
    assert used+max(0,p['maximum_new_artifact_allowance_bytes']-own)<99_990_000_000
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',
             PYTHONPATH=p['training_pythonpath'] if training else str(source),GIT_DIR='/home/ishaan/failure-detection/.git')
    print('START',name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/(name+'-process.json')).write_text(json.dumps(dict(pid=process.pid,command=command),indent=2)+'\n')
        assert process.wait()==0,name+' failed'
    print('DONE',name,flush=True)

verify();condition(status='training_compatibility_control',new_recovery_trials=0)
run('compatibility',p['training_commands']['compatibility'],True)
assert digest(root/'compatibility/boundary.npz')==p['compatibility_checkpoint_sha256']
models=['linear','nonlinear']
for name in models:
    condition(status='training_'+name,new_recovery_trials=0,compatibility_control_reproduced_exactly=True)
    run(name,p['training_commands'][name],True)
    trained=read(root/name/'training.json')
    assert trained['status']=='complete' and trained['base_weights_unchanged']
    assert trained['base_checkpoint_sha256']==p['base_checkpoint_sha256']
    assert trained['train_rows']==5064 and trained['validation_rows']==2040 and trained['samples_per_epoch']==3720
    assert len(trained['training_groups'])==130 and len(trained['validation_groups'])==36
    assert trained['extra_training']['sha256']==p['augmentation_sha256']
    assert trained['critical_material_training']['desired_critical_loss_mass']==.5
    assert trained['critical_material_training']['native_target_crop']==1
    assert trained['trained_parameters']==(10 if name=='linear' else 89)

import cv2
import numpy as np
from recovery import camera
from recovery.models.boundary import BoundarySegmenter
from recovery.parts import get_part
identity=[]
for case in p['identity_cases']:
    image=cv2.imread(case['image']);previous=cv2.imread(case['previous'])
    nominal=get_part(case['part']).layers();index=case['layer']
    for name in models:
        base=BoundarySegmenter(p['base_checkpoint']);model=BoundarySegmenter(root/name/'identity.npz')
        for m in [base,model]:m.prime_history(previous,camera.expected_mask(nominal[index-1]))
        expected=camera.expected_mask(nominal[index])
        a,b=base.signed_distance(image,expected),model.signed_distance(image,expected)
        assert np.array_equal(a,b),'identity head changes actual-image distances'
        ra,rb=base.measure(image,expected),model.measure(image,expected)
        assert ra.wkb==rb.wkb,'identity head changes actual-image geometry'
        identity.append(dict(case=case['name'],model=name,identity_sha256=model.sha256,
                             image_sha256=digest(case['image']),previous_sha256=digest(case['previous']),
                             distance_sha256=hashlib.sha256(a.tobytes()).hexdigest(),region_sha256=hashlib.sha256(ra.wkb).hexdigest()))
(root/'identity-verification.json').write_text(json.dumps(dict(status='verified',rows=identity),indent=2)+'\n')

condition(status='complete_print_camera_diagnostics',new_recovery_trials=0,identity_checks=len(identity))
diagnostics=[]
for job in p['diagnostics']:
    run(job['name'],job['command'])
    out=Path(job['out']);reference=Path(p['base_diagnostics'])/job['name']
    manifest=read(out/'manifest.json');final=read(out/'final.json');old=read(reference/'final.json')
    assert manifest['status']=='complete' and manifest['frames']==160 and final['truth']['ok']
    assert final['truth']==old['truth'] and final['measured']['classical']==old['measured']['classical']
    assert final['measured']['base']==old['measured']['combined']
    assert final['measured']['base:classical-base']==old['measured']['combined:classical-base']
    layers,old_layers=read(out/'layers.json'),read(reference/'layers.json')
    assert len(layers)==len(old_layers)==160
    for now,before in zip(layers,old_layers):
        assert all(now[k]==before[k] for k in ['layer','z','truth'])
        for label,old_label in [('classical','classical'),('base','combined'),('base:classical-base','combined:classical-base')]:
            assert now['measured'][label]==before['measured'][old_label]
    diagnostics.append(dict(case=job['case'],passes={name:final['measured'][name+':classical-base']['ok'] for name in models},
        final=final,manifest_sha256=digest(out/'manifest.json'),final_sha256=digest(out/'final.json'),
        layers_sha256=digest(out/'layers.json'),reference_layers_sha256=digest(reference/'layers.json'),
        all_160_control_feature_records_reproduced=True))
    (root/'diagnostic-results.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
qualified=[name for name in models if all(row['passes'][name] for row in diagnostics)]
if not qualified:
    verify();condition(status='rejected',reason='neither trained head passes all complete-print diagnostics',
        diagnostic_frames=640,new_recovery_trials=0,compatibility_control_reproduced_exactly=True,
        all_640_base_control_feature_records_reproduced=True,diagnostics=diagnostics)
    sys.exit(0)

condition(status='broad_camera_validation',qualified=qualified,new_recovery_trials=0,diagnostic_frames=640)
command=p['geometry_command_prefix'].copy()
for name in qualified:command+=['--checkpoint',name+'='+str(root/name/'boundary.npz')]
run('geometry',command)
reference={(r['group'],r['profile'],r['layer']):r for r in map(json.loads,Path(p['reference_geometry_frames']).read_text().splitlines())}
rows=[json.loads(line) for line in (root/'geometry-validation/frames.jsonl').read_text().splitlines()]
assert len(rows)==2040
flags=['open_wall','hard_thin','base_problem','connection_problem']
fatal=lambda v:any(v.get(k,False) for k in flags)
parts={};counts={};components={};lost={name:[] for name in qualified}
for name in ['classical','original','base',*qualified]:
    count=Counter();by_flag={key:Counter() for key in flags}
    for row in rows:
        ref=reference[(row['group'],row['profile'],row['layer'])]
        assert row['truth']==ref['truth'] and row['measured']['classical']==ref['measured']['classical']
        assert row['measured']['original']==ref['measured']['original']
        part_name=row['group'].split('-')[1]
        if part_name not in parts:
            part=get_part(part_name);parts[part_name]=(part,part.layers())
        part,nominal=parts[part_name]
        is_base=nominal[row['layer']].z<=part.spec['container']['base_h']
        measured=row['measured']['classical' if name in ['base',*qualified] and is_base else name]
        truth,predicted=fatal(row['truth']),fatal(measured)
        key=lambda t,m:('true_positive' if t else 'false_positive') if m else ('false_negative' if t else 'true_negative')
        count[key(truth,predicted)]+=1
        for flag in flags:by_flag[flag][key(row['truth'].get(flag,False),measured.get(flag,False))]+=1
        if name in qualified and truth and fatal(row['measured']['original']) and not predicted:
            lost[name].append([row['group'],row['profile'],row['layer']])
    counts[name]=dict(count);components[name]={flag:dict(value) for flag,value in by_flag.items()}
passes={name:not lost[name] and counts[name].get('false_negative',0)<=counts['original'].get('false_negative',0) for name in qualified}
verify();condition(status='camera_gates_passed' if any(passes.values()) else 'rejected',new_recovery_trials=0,
    diagnostic_frames=640,validation_frames=2040,qualified_heads=qualified,geometry_gate=passes,
    all_prior_geometry_controls_reproduced=True,all_640_base_control_feature_records_reproduced=True,
    functional_detection_counts=counts,component_detection_counts=components,
    lost_original_true_detections=lost,limitation=p['limitation'])
