from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess,sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root=base/'neural-native-coverage-v58';source=root/'source';prior=base/'neural-coverage-weighting-v57'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)]).split()[0])
used=size('/mnt/data/ishaan');assert used+16_000_000<99_990_000_000,used
assert (prior/'run.exit').read_text().strip()=='0' and read(prior/'condition.json')['status']=='rejected'
root.mkdir();source.mkdir()
for pattern in ['recovery/**/*.py','tools/*.py','parts/*.yaml']:
    for path in repo.glob(pattern):
        destination=source/path.relative_to(repo);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
shutil.copy2('/tmp/native-coverage-v58-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
py='/home/ishaan/failure-detection/.venv/bin/python'
old=read(base/'neural-nonlinear-refinement-v35/protocol.json')
compatibility=old['train_commands']['control'].copy()
compatibility[1]=str(source/'tools/train_boundary_refinement.py')
compatibility[compatibility.index('--out')+1]=str(root/'compatibility')
training={'compatibility':compatibility}
augmentation=base/'neural-training-coverage-v56/dataset/data.npz'
labels=prior/'dataset/critical-material.npz';checkpoint=prior/'candidate/boundary.npz'
for name,hidden in [('linear',0),('nonlinear',8)]:
    command=compatibility.copy()
    command[command.index('--out')+1]=str(root/name)
    command[command.index('--base-checkpoint')+1]=str(checkpoint)
    command[command.index('--hidden-channels')+1]=str(hidden)
    command+=['--extra-training-data',str(augmentation),'--extra-training-fraction','.5',
              '--critical-material-labels',str(labels),'--critical-pixel-mass','.5']
    training[name]=command
diagnostics=[];identity=[]
for oldjob in read(prior/'protocol.json')['diagnostics']:
    name=oldjob['name'];profile,part,kind,seed=oldjob['case'];out=root/name
    command=[py,str(source/'tools/inspect_camera_case.py'),'--profile',profile,'--part',part,'--kind',kind,'--seed',str(seed)]
    for label,path in [('base',checkpoint),('linear',root/'linear/boundary.npz'),('nonlinear',root/'nonlinear/boundary.npz')]:
        command+=['--checkpoint',label+'='+str(path)]
    command+=['--include-classical-base-hybrids','--out',str(out)]
    diagnostics.append(dict(name=name,case=oldjob['case'],out=str(out),command=command))
    image=base/'neural-complementary-cameras-v52'/(name+'-current.png')
    previous=image.with_name(name+'-previous.png')
    assert image.exists() and previous.exists()
    identity.append(dict(name=name,part=part,layer=45 if part=='mug' else 85,image=str(image),previous=str(previous)))
collection=base/'neural-context-camera-v8/trajectories/trajectories'
geometry=[py,str(source/'tools/validate_camera_geometry.py'),'--collections',str(collection),
    '--validation-seeds','70','71','72','73','400','410','417','420','--expected-groups','36',
    '--checkpoint','original='+str(base/'neural-boundary-v1/model/boundary.npz'),
    '--checkpoint','base='+str(checkpoint),'--workers','2','--out',str(root/'geometry-validation')]
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',source=str(source),
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    purpose='Learn native-grid spatial consistency after broader weighted training closes false gaps but retains false hard-thin alarms',
    base_checkpoint=str(checkpoint),base_checkpoint_sha256=digest(checkpoint),base_diagnostics=str(prior),
    compatibility_checkpoint_sha256=old['control_checkpoint_sha256'],
    augmentation_sha256=digest(augmentation),training_commands=training,
    training_pythonpath=str(base/'model-test-py311'),identity_cases=identity,diagnostics=diagnostics,
    geometry_command_prefix=geometry,reference_geometry_frames=str(base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl'),
    training_protocol='Freeze every base camera weight. Train a 10-parameter linear 3x3 head and an 89-parameter nonlinear residual head, identity-initialized, on the exact existing original-plus-augmented data and critical-material labels. Same 40 epochs, seed 2026, 3720 samples per epoch, 50/50 old/augmented sampling, and 50% critical-pixel loss mass. No new runtime input or contour repair. First reproduce the original 41bebe51 linear refinement checkpoint with all new options disabled.',
    weights_selected_by='Unweighted distance objective on the same 2040 original validation tiles, using the valid 62x62 native interior required by the 3x3 head. Identity and trained head are scored at identical coordinates; do not compare cropped MAE directly with prior whole-tile MAE.',
    identity_gate='Both identity exports must give exactly equal distances and WKB geometry to the base on all four retained onset images.',
    diagnostic_gate='For each head independently, all four complete-print diagnostics must pass the unchanged full specification using learned walls and classical specified bases. Verify all 640 per-layer base-control features, IoU/boundary error and final verdicts against v57. Failed heads do not advance.',
    geometry_gate='For each qualifying head, preserve every original true immediate functional-defect detection on all 2040 retained validation frames and do not increase combined false negatives. Report every component and false-positive count. Original truth, classical and original camera records must reproduce exactly.',
    maximum_new_artifact_allowance_bytes=16_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=0,
    reserved_fresh_seeds_untouched=[2001,2002,2003,2004],
    tests=dict(repository='231 passed, 3 skipped in 28.44s',torch_and_training='41 passed in 1.29s'),
    limitation='Training and camera qualification only. No recovery trial, final benchmark or promotion is automatic. A camera qualification result is not a completed-print recovery count.')
dataset=base/'neural-command-camera-v15/dataset'
pins=[root/'run.py',dataset/'data.npz',dataset/'manifest.json',dataset/'sampling-audit.json',augmentation,
    augmentation.parent/'manifest.json',labels,labels.parent/'manifest.json',checkpoint,checkpoint.parent/'training.json',
    base/'neural-command-camera-v15/command/boundary.npz',base/'neural-command-camera-v15/command/training.json',
    Path(old['control_checkpoint']),Path(old['control_training']),prior/'condition.json',prior/'critical-pixel-audit.json',
    base/'neural-boundary-v1/model/boundary.npz',base/'neural-boundary-v1/model/training.json',Path(p['reference_geometry_frames']),
    *[source/'tools'/name for name in ['train_boundary_refinement.py','train_boundary_tiles.py','boundary_training_augmentation.py',
       'boundary_critical_material.py','boundary_tiles.py','inspect_camera_case.py','validate_camera_geometry.py']]]
for case in identity:pins += [Path(case['image']),Path(case['previous'])]
for job in diagnostics:pins += [prior/job['name']/name for name in ['manifest.json','layers.json','final.json']]
p['pins']={str(path):digest(path) for path in pins}
assert p['source_sha256']==runtime_digest(repo)
assert p['simulation_contract_sha256']=='fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([py,'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','native-coverage-v58','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,source_sha256=p['source_sha256'])))
