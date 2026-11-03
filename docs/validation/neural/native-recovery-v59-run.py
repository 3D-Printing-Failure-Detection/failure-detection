from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,subprocess,sys

root=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source']);sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
from tools.result_storage import load_run_bytes
from recovery.eval.camera_trace import validate_camera_prefixes

def verify():
    assert runtime_digest(source)==p['source_sha256']
    assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
    for path,expected in p['pins'].items():assert digest(path)==expected,path

def save(**value):
    (root/'condition.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)

def execute(job):
    size=lambda q:int(subprocess.check_output(['du','-sx','-B1',str(q)]).split()[0])
    used,own=size('/mnt/data/ishaan'),size(root)
    assert own<p['maximum_new_artifact_allowance_bytes']
    assert used+max(0,p['maximum_new_artifact_allowance_bytes']-own)+1_000_000<99_990_000_000
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',
        PYTHONPATH=str(source),GIT_DIR='/home/ishaan/failure-detection/.git')
    print('START',job['name'],flush=True)
    with (root/(job['name']+'.log')).open('w') as log:
        process=subprocess.Popen(job['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/(job['name']+'-process.json')).write_text(json.dumps(dict(pid=process.pid,command=job['command']),indent=2)+'\n')
        assert process.wait()==0,job['name']
    directory=root/job['name']/'results'
    with (root/(job['name']+'-audit.log')).open('w') as log:
        subprocess.run([sys.executable,str(source/'tools/audit_neural_experiment.py'),str(directory),
            '--allow-unreached-agent','--require-camera-traces','--out',str(directory.parent/'audit')],
            cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=read(directory/'manifest.json');audit=read(directory.parent/'audit/audit.json')
    raw=load_run_bytes(directory)
    assert audit['status']=='passed' and audit['runs_sha256']==hashlib.sha256(raw).hexdigest()
    assert manifest['source_sha256']==manifest['final_source_sha256']==p['source_sha256']
    assert manifest['completed_runs']==job['expected_rows']
    assert manifest['boundary_checkpoint_sha256']==p['models'][job['model']]['sha256']
    assert manifest['arguments']['boundary_classical_base']==(job['model']!='control')
    assert manifest['arguments']['boundary_trend_continuation_fallback']
    assert not manifest['arguments']['contour_trend_shrinkage']
    rows=[json.loads(line) for line in raw.splitlines()]
    assert all(row['adaptations']<=3 and not row['clear_failure_on_adapt'] for row in rows)
    print('AUDITED',job['name'],flush=True)
    return rows

def batches(jobs):
    with ThreadPoolExecutor(max_workers=2) as executor:return list(executor.map(execute,jobs))

key=lambda row:tuple(row[k] for k in ['profile','part','kind','seed','method'])
fields=['profile','part','kind','seed','method','in_place','defective_finished','adaptations','layers_printed',
    'boundary_queries','neural_queries','agent_requests','damped_continuation_checks','damped_continuation_queries']
compact=lambda row:{k:row[k] for k in fields}
verify();save(status='paired_targeted_recovery',heads=p['heads'],new_rows=0)
jobs=p['target_jobs'];batches_done=batches(jobs)
rows_by_model={name:[row for job,rows in zip(jobs,batches_done) if job['model']==name for row in rows]
    for name in ['control',*p['heads']]}
control={key(row):row for row in rows_by_model['control']}
cases={tuple(case) for case in p['target_cases']}
assert set(control)=={case+(method,) for case in cases for method in ['ignore',*p['methods']]}
assert all(control[case+('ignore',)]['in_place'] for case in cases),'ignore reference no longer completes targets'
passes={};summary={}
for name in p['heads']:
    rows=rows_by_model[name]
    assert {key(row) for row in rows}=={case+(method,) for case in cases for method in p['methods']}
    for row in rows:
        for field in ['k0','severity','n_layers','clear_failure_on_adapt']:
            assert row[field]==control[key(row)][field]
    passes[name]=all(row['in_place'] and not row['defective_finished'] for row in rows)
    passes[name]&=sum(row['agent_requests'] for row in rows)>0
    passes[name]&=sum(row['boundary_queries'] for row in rows)>0
    summary[name]=[compact(row) for row in rows]
all_rows=[row for rows in rows_by_model.values() for row in rows]
paired=dict(passes=passes,control=[compact(row) for row in rows_by_model['control']],candidate=summary,
    camera_consistency=validate_camera_prefixes(all_rows,required=True),limitation=p['limitation'])
(root/'paired-comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
qualified=[name for name in p['heads'] if passes[name]]
if not qualified:
    verify();save(status='rejected',paired_gate=passes,new_rows=len(all_rows),preservation_rows=0)
    sys.exit(0)

save(status='preservation_recovery',paired_gate=passes,qualified=qualified,new_rows=len(all_rows))
reference={}
for directory in map(Path,p['reference_directories']):
    raw=load_run_bytes(directory);audit=read(directory.parent/'audit/audit.json')
    assert audit['status']=='passed' and audit['runs_sha256']==hashlib.sha256(raw).hexdigest()
    for line in raw.splitlines():
        row=json.loads(line);assert key(row) not in reference;reference[key(row)]=row
assert set(reference)=={tuple(case)+(method,) for case in p['preservation_cases'] for method in p['methods']}
assert len(reference)==50
jobs=[job for job in p['preservation_jobs'] if job['model'] in qualified]
done=batches(jobs);results={}
for name in qualified:
    rows=[row for job,items in zip(jobs,done) if job['model']==name for row in items]
    assert {key(row) for row in rows}==set(reference)
    for row in rows:
        for field in ['k0','severity','n_layers','clear_failure_on_adapt']:
            assert row[field]==reference[key(row)][field]
    methods={}
    for method in p['methods']:
        selected=[row for row in rows if row['method']==method]
        faults=[row for row in selected if row['kind']!='clean'];clean=[row for row in selected if row['kind']=='clean']
        lost=[list(key(row)) for row in selected if reference[key(row)]['in_place'] and not row['in_place']]
        added=[list(key(row)) for row in selected if row['defective_finished'] and not reference[key(row)]['defective_finished']]
        result=dict(fault_cases=len(faults),fault_successes=sum(row['in_place'] for row in faults),
            defective_finishes=sum(row['defective_finished'] for row in faults),clean_cases=len(clean),
            clean_successes=sum(row['in_place'] for row in clean),lost_reference_successes=lost,new_defects=added,
            neural_queries=sum(row['neural_queries'] for row in selected),boundary_queries=sum(row['boundary_queries'] for row in selected),
            agent_requests=sum(row['agent_requests'] for row in selected))
        result['passes']=bool(not lost and not added and result['fault_successes']>=14 and result['defective_finishes']<=1
            and result['clean_successes']==len(clean)==6 and result['neural_queries']>0 and result['boundary_queries']>0
            and (not method.endswith('_agent') or result['agent_requests']>0))
        methods[method]=result
    results[name]=dict(methods=methods,passes=all(row['passes'] for row in methods.values()),
        camera_consistency=validate_camera_prefixes(list(reference.values())+rows,required=True))
    all_rows+=rows
report=dict(status='passed' if any(value['passes'] for value in results.values()) else 'rejected',
    new_rows=len(all_rows),targeted=paired,preservation=results,source_sha256=p['source_sha256'],
    camera_consistency=validate_camera_prefixes(all_rows,required=True),limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
verify();save(status=report['status'],new_rows=len(all_rows),comparison_sha256=digest(root/'comparison.json'))
