from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess,sys,urllib.request

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
camera=base/'neural-native-coverage-v58';previous=base/'neural-continuation-fallback-v51';root=base/'neural-native-recovery-v59'
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert (camera/'run.exit').read_text().strip()=='0'
gate=read(camera/'condition.json');assert gate['status']=='camera_gates_passed'
heads=[name for name in ['linear','nonlinear'] if gate['geometry_gate'].get(name,False)]
assert heads
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)]).split()[0])
used=size('/mnt/data/ishaan');assert used+19_000_000<99_990_000_000,used
old=read(previous/'protocol.json');source=Path(read(camera/'protocol.json')['source'])
sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
supervisor_path=base/'local-supervisor-worker-v1/manifest.json';supervisor=read(supervisor_path)
actual=Path('/proc/2524739/cmdline').read_bytes().decode().strip('\0').split('\0')
assert actual==supervisor['server_command']
with Path(supervisor['reuses_model_file']).open('rb') as f:model_hash=hashlib.file_digest(f,'sha256').hexdigest()
assert model_hash==supervisor['verified_model_sha256']
with urllib.request.urlopen('http://127.0.0.1:8090/health',timeout=10) as response:health=json.load(response)
assert health['status']=='ok'
root.mkdir()
shutil.copy2('/tmp/native-recovery-v59-run.py',root/'run.py');shutil.copy2(__file__,root/'prepare.py')
attestation=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=2524739,
    server_command=actual,health=health,model_sha256=model_hash,manifest_sha256=digest(supervisor_path))
(root/'supervisor-runtime.json').write_text(json.dumps(attestation,indent=2)+'\n')
common=old['paired_jobs'][0]['command'];common=common[:common.index('--methods')]+['--boundary-trend-continuation-fallback']
models={'control':dict(path=str(base/'neural-boundary-v1/model/boundary.npz'))}
models.update({name:dict(path=str(camera/name/'boundary.npz')) for name in heads})
for value in models.values():value['sha256']=digest(value['path'])
target_jobs=[];target_cases=[]
groups=[('mugs',['cartesian','corexy'],'mug',1003),('cups',['cartesian','bedslinger'],'cup',1002)]
for group,profiles,part,seed in groups:
    target_cases += [[profile,part,'under_extrusion',seed] for profile in profiles]
    for name in models:
        label=name+'-'+group;command=common.copy()
        command[command.index('--boundary-checkpoint')+1]=models[name]['path']
        if name!='control':command+=['--boundary-classical-base']
        methods=(['ignore'] if name=='control' else [])+old['methods']
        command+=['--methods',*methods,'--n','1','--workers','2','--profiles',*profiles,'--parts',part,
            '--kinds','under_extrusion','--seed',str(seed),'--out',str(root/label/'results')]
        target_jobs.append(dict(name=label,model=name,command=command,expected_rows=len(methods)*len(profiles)))
reference_jobs=[job for job in old['paired_jobs'] if job['enabled']]+old['regression_jobs']
reference_directories=[str(previous/job['name']/'results') for job in reference_jobs]
preservation_jobs=[]
for name in heads:
    for job in reference_jobs:
        label=name+'-'+job['name'];command=job['command'].copy()
        command[command.index('--out')+1]=str(root/label/'results')
        command[command.index('--boundary-checkpoint')+1]=models[name]['path']
        command+=['--boundary-classical-base']
        preservation_jobs.append(dict(name=label,model=name,command=command,expected_rows=job['expected_rows']))
pins=[root/'run.py',root/'supervisor-runtime.json',supervisor_path,camera/'condition.json',camera/'protocol.json',
    camera/'run.exit',previous/'condition.json',previous/'comparison.json',
    base/'neural-equivariant-v1/classical/temporal.npz',base/'neural-equivariant-v1/classical/training.json']
for value in models.values():pins += [Path(value['path']),Path(value['path']).parent/'training.json']
for directory in map(Path,reference_directories):
    pins += [directory/'manifest.json',directory.parent/'audit/audit.json']
    pins += [directory/'runs.jsonl'] if (directory/'runs.jsonl').exists() else [directory/'runs.jsonl.gz',directory/'runs-storage.json']
pins += [source/'tools'/name for name in ['audit_neural_experiment.py','experiment_provenance.py','result_storage.py']]
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',source=str(source),
    source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    heads=heads,models=models,methods=old['methods'],target_jobs=target_jobs,target_cases=target_cases,
    preservation_jobs=preservation_jobs,preservation_cases=old['paired_cases']+old['regression_cases'],
    reference_directories=reference_directories,pins={str(path):digest(path) for path in pins},
    maximum_new_artifact_allowance_bytes=18_000_000,supervisor_log_reservation_bytes=1_000_000,
    storage_bytes_at_launch=used,concurrent_commands=2,workers_per_command=2,
    purpose='Test camera-qualified refinements inside the existing neural/Qwen continuation-fallback policy, using specified classical bases.',
    acceptance='Both neural and actual-Qwen variants must finish all four target cases without defects, matching successful ignore controls. Each passing head then preserves every success of the 50 v51 candidate/preservation rows, with no new defective case, all six clean controls per variant, and actual temporal, camera and Qwen inference. Keep all source/physics/model pins, failure persistence, three-adaptation limit, and matching common camera frames.',
    reserved_fresh_seeds_untouched=[2001,2002,2003,2004],
    limitation='Consumed targeted development and preservation only. Camera-error avoidance is not a rescue of an otherwise defective print. No full-grid or fresh-test superiority is claimed; no promotion or broad run is automatic.')
assert p['source_sha256']==old['source_sha256']
assert p['simulation_contract_sha256']==old['simulation_contract_sha256']
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+shlex.quote(str(source))+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch+=shlex.join([str(repo/'.venv/bin/python'),'-u',str(root/'run.py')])+' > '+shlex.quote(str(root/'run.log'))+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+shlex.quote(str(root/'run.exit'))+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','native-recovery-v59','-P','-F','#{window_id}','bash '+shlex.quote(str(root/'launch.sh'))],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json'),runner_sha256=digest(root/'run.py')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,heads=heads,maximum_rows=12+58*len(heads))))
