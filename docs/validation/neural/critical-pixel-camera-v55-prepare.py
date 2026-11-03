from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
oldroot=base/'neural-command-history-v50';root=base/'neural-critical-pixel-camera-v55';source=root/'source'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)],text=True).split()[0])
used=size('/mnt/data/ishaan');other=base/'neural-continuation-fallback-v51'
reserved=max(0,12_000_000-size(other))+1_000_000 if not (other/'run.exit').exists() else 0
assert used+reserved+8_000_000<99_990_000_000,(used,reserved)
root.mkdir();source.mkdir()
for pattern in ['recovery/**/*.py','tools/*.py','parts/*.yaml']:
    for path in repo.glob(pattern):
        destination=source/path.relative_to(repo);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
shutil.copy2('/tmp/critical-pixel-camera-v55-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
def relocate(value):
    if isinstance(value,str):return value.replace(str(oldroot),str(root))
    if isinstance(value,list):return [relocate(x) for x in value]
    if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
    return value
p=relocate(read(oldroot/'protocol.json'))
dataset=base/'neural-command-camera-v15/dataset';collection=base/'neural-context-camera-v8/trajectories/trajectories'
py='/home/ishaan/failure-detection/.venv/bin/python'
p['dataset_command']=[py,str(source/'tools/boundary_critical_material.py'),'--dataset',str(dataset),
    '--audit',str(dataset/'sampling-audit.json'),'--trajectories',str(collection),'--out',str(root/'dataset')]
command=p['training_commands']['candidate'];index=command.index('--previous-command-masks');del command[index:index+2]
command+=['--critical-material-labels',str(root/'dataset/critical-material.npz'),'--critical-pixel-mass','.5']
for job in p['diagnostics']:
    job['command']=[arg.replace('previous_command=','critical_material=') for arg in job['command']]
p['geometry_command']=[arg.replace('previous_command=','critical_material=') for arg in p['geometry_command']]
p.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    purpose='Test whether scarce true thin-material pixels are underweighted even after whole-tile rebalancing, without adding inputs or changing geometry checks',
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    maximum_new_artifact_allowance_bytes=8_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=reserved,
    training_protocol='Same three-channel inputs, 5760 tiles, 66/36 training/validation groups, 40 epochs, seed 2026, architecture, initialization, optimizer and critical-tile sampling. Reproduce the original command-camera checkpoint exactly. Candidate weights the audited 1648 true thin-material training pixels to half the expected distance-loss weight under the actual training sampler. Ordinary pixels retain the other half. Factors use training labels only. Validation selection remains the original unweighted whole-tile objective.',
    weights_selected_by='Unchanged full tile-validation loss; neither the four consumed prints nor critical-pixel recall selects an epoch',
    tests=dict(repository='221 passed, 3 skipped in 28.85s',torch_export_refinement_and_critical_labels='29 passed in 1.24s',critical_and_command_sidecars='12 passed in 0.26s'))
pins=[root/'run.py',dataset/'data.npz',dataset/'manifest.json',dataset/'sampling-audit.json',
    base/'neural-command-camera-v15/command/boundary.npz',base/'neural-command-camera-v15/command/training.json',
    base/'neural-boundary-v1/model/boundary.npz',base/'neural-boundary-v1/model/training.json',
    base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl',
    base/'neural-complementary-cameras-v52/comparison.json',base/'neural-mask-topology-v54/comparison.json',
    *[source/'tools'/name for name in ['boundary_critical_material.py','train_boundary_tiles.py',
                                    'validate_camera_geometry.py','inspect_camera_case.py']]]
p['pins']={str(path):digest(path) for path in pins}
assert p['source_sha256']==runtime_digest(repo)
assert p['simulation_contract_sha256']=='fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([py,'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','critical-pixels-v55','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,reserved=reserved,source_sha256=p['source_sha256'])))
