from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo=Path('/home/ishaan/failure-detection')
base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root=base/'neural-complementary-cameras-v52'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)],text=True).split()[0])
used=size('/mnt/data/ishaan'); ongoing=base/'neural-continuation-fallback-v51'
reserved=max(0,12_000_000-size(ongoing))+1_000_000 if not (ongoing/'run.exit').exists() else 0
assert used+reserved+4_000_000<99_990_000_000,(used,reserved)
old=read(base/'neural-camera-symmetry-v49/protocol.json')
source=Path(read(ongoing/'protocol.json')['source'])
sys.path.insert(0,str(source))
from recovery.models.boundary import BoundarySegmenter
from tools.experiment_provenance import runtime_digest,simulation_contract

unique={}
for checkpoint in sorted(base.glob('*/*/boundary.npz')):
    if not (checkpoint.parent/'training.json').exists():continue
    unique.setdefault(digest(checkpoint),[]).append(checkpoint)
assert len(unique)==26
models=[];pins={}
for sha,paths in unique.items():
    checkpoint=paths[0]
    model=BoundarySegmenter(checkpoint)
    assert model.sha256==sha
    name=str(checkpoint.parent.relative_to(base)).replace('/','--')
    models.append(dict(name=name,checkpoint=str(checkpoint),sha256=sha,
        aliases=[str(path) for path in paths],input_frames=model.input_frames,
        input_scales=model.input_scales,include_command=model.include_expected_mask,
        include_previous_command=model.include_previous_expected_mask))
    for path in paths:
        pins[str(path)]=sha
        pins[str(path.parent/'training.json')]=digest(path.parent/'training.json')
for path in [Path(case['reference_frames']) for case in old['consumed_cases']]+[
    Path(job['trajectory']) for job in old['real_defect_groups']]+[
    Path(old['validation_frames']),source/'tools/inspect_camera_case.py',source/'tools/validate_camera_geometry.py']:
    pins[str(path)]=digest(path)
root.mkdir()
shutil.copy2('/tmp/complementary-cameras-v52-run.py',root/'run.py')
shutil.copy2(__file__,root/'prepare.py')
pins[str(root/'run.py')]=digest(root/'run.py')
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',
    source=str(source),source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    purpose='Determine whether already-trained camera models provide complementary full-spec-compatible measurements on known false alarms, while retaining selected real defect detections',
    models=models,consumed_cases=old['consumed_cases'],real_defect_groups=old['real_defect_groups'],
    validation_frames=old['validation_frames'],pins=pins,expected_models=26,expected_frames=22,
    expected_predictions=572,maximum_new_artifact_allowance_bytes=4_000_000,
    storage_bytes_at_launch=used,other_live_reserved_bytes=reserved,
    procedure='Evaluate every unique retained checkpoint with training provenance on identical actual current/previous images and issued command masks. Reproduce prior nonlinear features and truth. Score all four consumed local windows, all twelve support edges and six selected true defects. No averaging, per-pixel switching, threshold fitting or geometry modification.',
    screening='A model covers a local case only if all four frames have no immediate fatal violation, all three support edges stay at or below the unchanged 6 mm2 limit, and all six selected real defects remain detected. Report every model and case; any useful coverage only qualifies separate complete-print and broad validation.',
    limitation='Exploratory consumed-development screening of retained trained models. Local coverage is not complete-print validity, full validation or recovery success. The six true defects are selected correlated frames. Existing failed experiments remain failed; no model is promoted, training repeated or fresh seeds used.')
assert p['simulation_contract_sha256']==old['simulation_contract_sha256']
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([str(repo/'.venv/bin/python'),'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','camera-complements-v52','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,models=len(models),used=used,reserved=reserved)))
