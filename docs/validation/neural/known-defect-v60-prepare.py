from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess,sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
camera=base/'neural-native-coverage-v58';root=base/'neural-known-defect-v60'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
usage=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)]).split()[0])
used=usage('/mnt/data/ishaan');old=read(camera/'protocol.json');source=Path(old['source'])
remaining=max(0,old['maximum_new_artifact_allowance_bytes']-usage(camera))
assert used+remaining+19_000_000+4_000_000<99_990_000_000
assert not root.exists()
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
from tools.result_storage import load_run_bytes
from recovery.deform import random_failure
from recovery.parts import get_part
import numpy as np
case=['bedslinger','mug','under_extrusion',1004]
report_path=base/'neural-trend-full-v38/comparison.json';report=read(report_path);ignore=None
for entry in report['inputs']:
    directory=Path(entry['directory']);manifest=read(directory/'manifest.json')
    methods=[method for method in manifest['arguments']['methods'] if entry['labels'].get(method,method)=='ignore']
    if not methods:continue
    raw=load_run_bytes(directory)
    assert hashlib.sha256(raw).hexdigest()==entry['runs_sha256']
    for row in map(json.loads,raw.splitlines()):
        if row['method'] in methods and [row[k] for k in ['profile','part','kind','seed']]==case:
            assert ignore is None;ignore=row
assert ignore and ignore['defective_finished'] and ignore['adaptations']==0
part=get_part('mug');failure=random_failure(part.layers(),np.random.default_rng(1004),kind='under_extrusion')
assert failure.k0==ignore['k0'] and failure.p['severity']==ignore['severity']
prior=base/'neural-continuation-fallback-v51/defects-1004/results'
assert read(prior.parent/'audit/audit.json')['runs_sha256']==hashlib.sha256(load_run_bytes(prior)).hexdigest()
checkpoints={'original':base/'neural-boundary-v1/model/boundary.npz',
    'base':base/'neural-coverage-weighting-v57/candidate/boundary.npz',
    'linear':camera/'linear/boundary.npz','nonlinear':camera/'nonlinear/boundary.npz'}
arguments=['inspect','--profile','bedslinger','--part','mug','--kind','under_extrusion','--seed','1004']
for name,path in checkpoints.items():arguments+=['--checkpoint',name+'='+str(path)]
arguments+=['--include-classical-base-hybrids','--snapshot-layers','100','101','102','--out',str(root/'diagnostic')]
root.mkdir();shutil.copy2('/tmp/known-defect-v60-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
pins=[root/'run.py',report_path,prior/'manifest.json',prior.parent/'audit/audit.json',
      source/'tools/inspect_camera_case.py',source/'tools/experiment_provenance.py',source/'tools/result_storage.py']
pins += [prior/'runs.jsonl'] if (prior/'runs.jsonl').exists() else [prior/'runs.jsonl.gz',prior/'runs-storage.json']
for path in checkpoints.values():pins += [path,path.parent/'training.json']
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    case=case,prior_results=str(prior),ignore_violations=ignore['violations'],
    paired_failure=dict(k0=ignore['k0'],severity=ignore['severity']),
    diagnostic_arguments=arguments,checkpoint_sha256={name:digest(path) for name,path in checkpoints.items()},
    pins={str(path):digest(path) for path in pins},maximum_new_artifact_allowance_bytes=4_000_000,
    storage_bytes_at_launch=used,other_live_camera_reservation=remaining,pending_recovery_reservation=19_000_000,
    purpose='Determine whether new camera measurements recognize the real wall defect in the already consumed mug case that both ignore and the prior controller finish defective.',
    limitation='Supplemental camera diagnosis on one consumed nominal-command trajectory. No recovery controller runs, no physical or scoring rule changes, no checkpoint selection, and no replacement for the ongoing 2040-frame or queued recovery gates.')
assert p['source_sha256']==old['source_sha256'] and p['simulation_contract_sha256']==old['simulation_contract_sha256']
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+='export PYTHONPATH='+shlex.quote(str(source))+' GIT_DIR=/home/ishaan/failure-detection/.git\n'
launch+=shlex.join([str(repo/'.venv/bin/python'),'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','known-defect-v60','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(window=window,protocol_sha256=digest(root/'protocol.json')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,storage_bytes=used,other_reserved=remaining+19_000_000)))
