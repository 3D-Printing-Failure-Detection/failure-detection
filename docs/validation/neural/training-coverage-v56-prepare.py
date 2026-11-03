from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
oldroot=base/'neural-critical-pixel-camera-v55';root=base/'neural-training-coverage-v56';source=root/'source'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)],text=True).split()[0])
used=size('/mnt/data/ishaan')
assert used+28_000_000<99_990_000_000,used
assert read(base/'neural-continuation-fallback-v51/condition.json')['status']=='passed'
assert (base/'neural-continuation-fallback-v51/run.exit').read_text().strip()=='0'
assert read(oldroot/'condition.json')['status']=='rejected' and (oldroot/'run.exit').read_text().strip()=='0'
root.mkdir();source.mkdir()
for pattern in ['recovery/**/*.py','tools/*.py','parts/*.yaml']:
    for path in repo.glob(pattern):
        destination=source/path.relative_to(repo);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
shutil.copy2('/tmp/training-coverage-v56-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
def relocate(value):
    if isinstance(value,str):return value.replace(str(oldroot),str(root))
    if isinstance(value,list):return [relocate(x) for x in value]
    if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
    return value
p=relocate(read(oldroot/'protocol.json'))
dataset=base/'neural-command-camera-v15/dataset';collection=base/'neural-context-camera-v8/trajectories/trajectories'
py='/home/ishaan/failure-detection/.venv/bin/python'
p['dataset_command']=[py,str(source/'tools/boundary_training_augmentation.py'),'--dataset',str(dataset),
    '--trajectories',str(collection),'--out',str(root/'dataset'),'--seeds',*[str(i) for i in range(500,508)],
    '--severities','.90','1.05','1.15','1.30','--workers','2']
command=p['training_commands']['candidate']
for flag in ['--critical-material-labels','--critical-pixel-mass']:
    index=command.index(flag);del command[index:index+2]
command+=['--extra-training-data',str(root/'dataset/data.npz'),'--extra-training-fraction','.5']
for job in p['diagnostics']:
    job['command']=[arg.replace('critical_material=','coverage=') for arg in job['command']]
p['geometry_command']=[arg.replace('critical_material=','coverage=') for arg in p['geometry_command']]
p.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    purpose='Test broader training-only coverage after rare-pixel weighting overfit two trajectories',
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    maximum_new_artifact_allowance_bytes=28_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=0,
    augmentation_protocol='64 groups: cup and mug, eight new training seeds 500-507, and four severities 0.90, 1.05, 1.15, 1.30 mm within the unchanged fault range. Apply the original failure operator to retained clean training deposits, render all three existing camera profiles, and sample four command-boundary locations at the first and last affected frame. Retain true surviving walls and true gaps. This is geometry-based training augmentation, not new physical rollouts. Original validation and diagnostic images are unchanged.',
    coverage_gate='At least eight groups with >=16 audited thin-positive pixels, no more than half of those pixels from the two largest groups, and at least eight groups with true open-wall frames.',
    training_protocol='Reproduce the original command-camera control exactly. Same three inputs, model, initialization, 40 epochs, optimizer and 3720 samples per epoch. Candidate assigns half sampling mass to original training rows using their existing weights and half uniformly across new groups then their tiles. No per-pixel reweighting. Original 2040 validation tiles and selection objective remain unchanged.',
    weights_selected_by='Unchanged original full tile-validation objective; diagnostic cases and retrospective rare-pixel recall never select an epoch',
    tests=dict(repository='230 passed, 3 skipped in 28.29s',torch_export_refinement_labels_and_augmentation='38 passed in 1.26s',augmentation_and_labels='18 passed in 0.28s'))
pins=[root/'run.py',dataset/'data.npz',dataset/'manifest.json',dataset/'sampling-audit.json',
    base/'neural-command-camera-v15/command/boundary.npz',base/'neural-command-camera-v15/command/training.json',
    base/'neural-boundary-v1/model/boundary.npz',base/'neural-boundary-v1/model/training.json',
    base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl',
    oldroot/'critical-pixel-audit-verified.json',oldroot/'condition.json',
    base/'neural-continuation-fallback-v51/comparison.json',
    *[source/'tools'/name for name in ['boundary_training_augmentation.py','boundary_critical_material.py',
                                    'boundary_tiles.py','train_boundary_tiles.py','validate_camera_geometry.py','inspect_camera_case.py']]]
p['pins']={str(path):digest(path) for path in pins}
assert p['source_sha256']==runtime_digest(repo)
assert p['simulation_contract_sha256']=='fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([py,'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','training-coverage-v56','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,source_sha256=p['source_sha256'])))
