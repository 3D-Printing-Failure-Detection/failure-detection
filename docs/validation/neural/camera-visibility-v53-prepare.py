from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root=base/'neural-camera-visibility-v53'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)],text=True).split()[0])
used=size('/mnt/data/ishaan')
reserved=sum(max(0,limit-size(base/name)) for name,limit in [
    ('neural-continuation-fallback-v51',12_000_000),('neural-complementary-cameras-v52',4_000_000)]
    if not (base/name/'run.exit').exists())+1_000_000
assert used+reserved+2_000_000<99_990_000_000,(used,reserved)
source=Path(read(base/'neural-continuation-fallback-v51/protocol.json')['source'])
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
import inspect,mujoco
renderer_source=Path(inspect.getfile(mujoco.Renderer))
old=read(base/'neural-camera-symmetry-v49/protocol.json')
cases=[dict(case,snapshot_layer=45 if case['part']=='mug' else 85)
       for case in old['consumed_cases'] if case['profile']=='cartesian']
root.mkdir();shutil.copy2('/tmp/camera-visibility-v53-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
reference=base/'neural-camera-symmetry-v49/frames.jsonl'
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',source=str(source),
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    purpose='Determine whether older rendered layers obscure the valid thin current layer in known false camera gaps',
    cases=cases,reference_frames=str(reference),
    pins={str(path):digest(path) for path in [reference,renderer_source,root/'run.py']},
    procedure='Reproduce the eight original actual images exactly. Label existing visual geoms for offline current-layer visibility, verify labels do not change ordinary RGB, and compare with a same-pose rendering that omits older visual layers. Keep all printed geometry, physical state, camera frame counts and camera RNG unchanged.',
    visibility_measure='Fraction of projected current-only visible pixels hidden in the complete scene, overall and within a dilated nominal/actual-difference region. Segmentation masks use the original calibrated projection and fixed half-intensity threshold. Also report cavity closure of true raster and visible current-layer masks.',
    maximum_new_artifact_allowance_bytes=2_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=reserved,
    limitation='Offline causal visualization only. Segmentation labels and current-only images are unavailable to runtime models and are not proposed as improved policy inputs. Results distinguish occlusion by older layers from other image/model limitations on these consumed frames only; no training, recovery trial, physics change or fresh seed.')
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([str(repo/'.venv/bin/python'),'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','camera-visibility-v53','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,reserved=reserved)))
