from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
base=Path('/mnt/data/ishaan/hri-workspace/failure-detection');repo=Path('/home/ishaan/failure-detection')
root=base/'neural-command-history-v50';source=root/'source'
sys.path.insert(0,str(repo))
from tools.experiment_provenance import runtime_digest,simulation_contract
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
usage=lambda p:int(subprocess.check_output(['du','-sx','-B1',str(p)]).split()[0])
used=usage('/mnt/data/ishaan');history=base/'neural-trend-historical-v43'
reserved=max(0,24_000_000-usage(history)) if not (history/'report-resume.exit').exists() else 0
assert used+reserved+12_000_000<99_990_000_000,(used,reserved)
assert not root.exists();root.mkdir();source.mkdir()
for pattern in ['recovery/**/*.py','tools/*.py','parts/*.yaml']:
    for path in repo.glob(pattern):
        target=source/path.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
shutil.copyfile('/tmp/command-history-v50-run.py',root/'run.py');shutil.copyfile(__file__,root/'prepare.py')
assert runtime_digest(source)==runtime_digest(repo)
assert simulation_contract(source)['sha256']=='fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
py='/home/ishaan/failure-detection/.venv/bin/python';torch='/mnt/data/ishaan/hri-workspace/envs/openpi/bin/python'
dataset=base/'neural-command-camera-v15/dataset';control=base/'neural-command-camera-v15/command'
commands={}
for name in ['control','candidate']:
    command=[torch,str(source/'tools/train_boundary_tiles.py'),'--data',str(dataset/'data.npz'),
        '--sampling-audit',str(dataset/'sampling-audit.json'),'--critical-tile-fraction','.5',
        '--include-expected-mask','--epochs','40','--batch-size','32','--seed','2026','--out',str(root/name)]
    if name=='candidate':command+=['--previous-command-masks',str(root/'dataset/previous-command-masks.npz')]
    commands[name]=command
diagnostics=[]
for profile,part,seed in [('cartesian','mug',1003),('corexy','mug',1003),('cartesian','cup',1002),('bedslinger','cup',1002)]:
    name=f'{profile}-{part}-{seed}';out=root/name
    command=[py,str(source/'tools/inspect_camera_case.py'),'--profile',profile,'--part',part,
        '--kind','under_extrusion','--seed',str(seed),'--checkpoint','control='+str(root/'control/boundary.npz'),
        '--checkpoint','previous_command='+str(root/'candidate/boundary.npz'),
        '--include-classical-base-hybrids','--out',str(out)]
    diagnostics.append(dict(name=name,case=[profile,part,'under_extrusion',seed],out=str(out),command=command))
collection=base/'neural-context-camera-v8/trajectories/trajectories'
original=base/'neural-boundary-v1/model/boundary.npz'
geometry=[py,str(source/'tools/validate_camera_geometry.py'),'--collections',str(collection),
    '--validation-seeds','70','71','72','73','400','410','417','420','--expected-groups','36',
    '--checkpoint','original='+str(original),'--checkpoint','control='+str(root/'control/boundary.npz'),
    '--checkpoint','previous_command='+str(root/'candidate/boundary.npz'),'--workers','2',
    '--out',str(root/'geometry-validation')]
pins=[root/'run.py',dataset/'data.npz',dataset/'manifest.json',dataset/'sampling-audit.json',
    control/'boundary.npz',control/'training.json',original,original.parent/'training.json',
    base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl',
    *[source/'tools'/name for name in ['add_boundary_command_history.py','train_boundary_tiles.py','validate_camera_geometry.py','inspect_camera_case.py']]]
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',
    purpose='Train camera measurements on the actual preceding command as well as the current command and two observed images, to distinguish expected layer transitions from defects',
    source_directory=str(source),source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    maximum_new_artifact_allowance_bytes=12_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=reserved,
    dataset_command=[py,str(source/'tools/add_boundary_command_history.py'),'--dataset',str(dataset),'--trajectories',str(collection),'--out',str(root/'dataset')],
    training_commands=commands,training_pythonpath=str(base/'model-test-py311'),diagnostics=diagnostics,geometry_command=geometry,
    reference_control_sha256=digest(control/'boundary.npz'),
    reference_geometry_frames=str(base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl'),
    training_protocol='Same 5760 camera tiles, 66 training and 36 validation groups, 40 epochs, seed 2026, loss, optimizer and critical-tile sampling. Original three-channel control must reproduce its exact checkpoint. Candidate adds one known command channel and 5184 input weights; no architecture-capacity attribution is claimed.',
    diagnostic_gate='All four complete unchanged-command prints must pass the unchanged full specification using learned walls and classical specified bases. Their actual ignore successes must reproduce.',
    geometry_gate='All 2040 retained validation images. Preserve every original true immediate functional-defect detection and do not increase combined false negatives; report every component metric and false positive. Old truth, classical and original-model frame records must match exactly. No truth enters inference.',
    weights_selected_by='Unchanged whole tile-validation loss, never the four consumed outcomes or functional-flag counts',
    reserved_fresh_seeds_untouched=[2001,2002,2003,2004],pins={str(path):digest(path) for path in pins},
    tests=dict(repository='201 passed, 3 skipped in 29.47s',torch_export_and_refinement='20 passed in 2.00s'),
    limitation='Training and selected/broad camera validation only. Passing camera gates does not establish closed-loop recovery, old-success preservation, agent benefit, historical regression or fresh-test superiority. No recovery jobs or automatic policy promotion are included.')
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+str(source)+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH='+str(source)+' GIT_DIR=/home/ishaan/failure-detection/.git\n'
launch+=py+' -u '+str(root/'run.py')+' > '+str(root/'run.log')+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+str(root/'run.exit')+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','command-history-v50','-P','-F','#{window_id}','bash '+str(root/'launch.sh')],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,reserved=reserved,source_sha256=p['source_sha256'])))
