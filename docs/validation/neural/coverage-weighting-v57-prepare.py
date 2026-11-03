from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess,sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
oldroot=base/'neural-training-coverage-v56';root=base/'neural-coverage-weighting-v57';source=root/'source'
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
size=lambda p:int(subprocess.check_output(['du','-sx','-B1',str(p)]).split()[0])
used=size('/mnt/data/ishaan');reserved=max(0,28_000_000-size(oldroot)) if not (oldroot/'run.exit').exists() else 0
assert used+reserved+8_000_000<99_990_000_000,(used,reserved)
assert read(oldroot/'candidate/training.json')['status']=='complete'
root.mkdir();source.mkdir()
for pattern in ['recovery/**/*.py','tools/*.py','parts/*.yaml']:
    for path in repo.glob(pattern):
        destination=source/path.relative_to(repo);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
shutil.copy2('/tmp/coverage-weighting-v57-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
def relocate(value):
    if isinstance(value,str):return value.replace(str(oldroot),str(root))
    if isinstance(value,list):return [relocate(x) for x in value]
    if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
    return value
p=relocate(read(oldroot/'protocol.json'))
dataset=base/'neural-command-camera-v15/dataset';labels=base/'neural-critical-pixel-camera-v55/dataset/critical-material.npz'
py='/home/ishaan/failure-detection/.venv/bin/python'
p['dataset_command']=[py,str(source/'tools/extend_boundary_critical_material.py'),'--original-data',str(dataset/'data.npz'),
    '--original-labels',str(labels),'--augmentation',str(oldroot/'dataset'),
    '--trajectories',str(base/'neural-context-camera-v8/trajectories/trajectories'),'--out',str(root/'dataset')]
command=p['training_commands']['candidate']
command[command.index('--extra-training-data')+1]=str(oldroot/'dataset/data.npz')
control=command.copy();control[control.index('--out')+1]=str(root/'control')
p['training_commands']['control']=control
command+=['--critical-material-labels',str(root/'dataset/critical-material.npz'),'--critical-pixel-mass','.5']
for job in p['diagnostics']:
    job['command']=[arg.replace('coverage=','combined=') for arg in job['command']]
p['geometry_command']=[arg.replace('coverage=','combined=') for arg in p['geometry_command']]
p.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    purpose='Complete the data-coverage and pixel-weighting comparison using exactly the retained augmented examples',
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    maximum_new_artifact_allowance_bytes=8_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=reserved,
    coverage_root=str(oldroot),coverage_runner_pid=3491846,
    reference_control_sha256=read(oldroot/'candidate/training.json')['checkpoint_sha256'],
    augmentation_protocol='No new examples or images: use the completed, SHA-bound 1344-tile coverage dataset unchanged. Reconstruct and verify every extra target and critical-material label using its original parent and fault definition. Bind the extended labels to both dataset hashes.',
    training_protocol='Both models use the same 5064 training rows, 2040 original validation rows, initialization, architecture, optimizer, 40 epochs and 3720 samples per epoch. Original/augmented sampling mass stays 50/50. Reproduce the unweighted coverage control exactly; only the candidate weights 31738 audited thin-positive training pixels to half expected distance-loss mass under that sampler. Validation selection is unchanged.',
    factorial_controls='Original data/unweighted: v15. Original data/weighted: v55. Augmented data/unweighted: v56, reproduced here. Augmented data/weighted: this candidate. Earlier failed outcomes stay retained; no factor is claimed to improve recovery from pixel scores.',
    tests=dict(repository='231 passed, 3 skipped in 28.72s',torch_export_refinement_labels_and_augmentation='39 passed in 1.32s',combined_label_tests='19 passed in 0.28s'))
p.pop('coverage_gate',None)
pins=[root/'run.py',dataset/'data.npz',dataset/'manifest.json',dataset/'sampling-audit.json',labels,
    oldroot/'dataset/data.npz',oldroot/'dataset/manifest.json',oldroot/'coverage-audit.json',
    oldroot/'candidate/boundary.npz',oldroot/'candidate/training.json',oldroot/'critical-pixel-audit.json',
    base/'neural-boundary-v1/model/boundary.npz',base/'neural-boundary-v1/model/training.json',
    base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl',
    *[source/'tools'/name for name in ['boundary_training_augmentation.py','boundary_critical_material.py',
      'extend_boundary_critical_material.py','boundary_tiles.py','train_boundary_tiles.py','validate_camera_geometry.py','inspect_camera_case.py']]]
p['pins']={str(path):digest(path) for path in pins}
assert p['source_sha256']==runtime_digest(repo)
assert p['simulation_contract_sha256']=='fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([py,'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','coverage-weighting-v57','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,reserved=reserved,source_sha256=p['source_sha256'])))
