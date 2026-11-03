"""Diagnose a consumed defective print; never run or alter a recovery policy."""
from pathlib import Path
import hashlib,json,sys

root=Path(__file__).resolve().parent
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source']);sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
from tools.result_storage import load_run_bytes
from recovery.eval.camera_trace import camera_frame_digest,decode_camera_prefix,encode_camera_prefix
from recovery.printer.mujoco import MujocoPrinter
from tools.inspect_camera_case import main

def verify():
    assert runtime_digest(source)==p['source_sha256']
    assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
    for path,sha in p['pins'].items():assert digest(path)==sha,path

verify()
traces=[];original_camera=MujocoPrinter.camera_image
def capture(printer):
    image=original_camera(printer)
    traces.append(camera_frame_digest(image))
    return image
MujocoPrinter.camera_image=capture
(root/'condition.json').write_text(json.dumps(dict(status='camera_diagnostic',recovery_trials=0),indent=2)+'\n')
sys.argv=p['diagnostic_arguments']
try:main()
finally:MujocoPrinter.camera_image=original_camera
out=root/'diagnostic';manifest=read(out/'manifest.json');final=read(out/'final.json')
assert manifest['status']=='complete' and manifest['frames']==len(traces)==160
assert manifest['source_sha256']==manifest['final_source_sha256']==p['source_sha256']
assert manifest['checkpoint_sha256']==p['checkpoint_sha256']
assert not final['truth']['ok'] and final['truth']['violations']==p['ignore_violations']
rows=[json.loads(line) for line in load_run_bytes(p['prior_results']).splitlines()]
matched=[]
for row in rows:
    if tuple(row[k] for k in ['profile','part','kind','seed'])!=tuple(p['case']):continue
    assert row['defective_finished'] and row['violations']==p['ignore_violations']
    trace=decode_camera_prefix(row,required=True)
    assert trace and tuple(traces[:len(trace)])==trace,'actual camera prefix changed'
    matched.append(dict(method=row['method'],frames=len(trace),
        first_adaptation=next(event['layer'] for event in row['actions'] if event['action']=='adapt')))
assert len(matched)==2
layers=read(out/'layers.json');flags=['open_wall','hard_thin','base_problem','connection_problem']
fatal=lambda value:any(value.get(flag,False) for flag in flags)
defects=[]
for row in layers:
    if fatal(row['truth']):
        defects.append(dict(layer=row['layer'],truth=row['truth'],measured=row['measured']))
assert defects
verify()
result=dict(status='complete',case=p['case'],recovery_trials=0,source_sha256=p['source_sha256'],
    all_pre_adaptation_camera_frames_reproduced=True,paired_prefixes=matched,
    camera_prefix=encode_camera_prefix(traces),true_fatal_layers=defects,final=final,
    manifest_sha256=digest(out/'manifest.json'),layers_sha256=digest(out/'layers.json'),
    final_sha256=digest(out/'final.json'),limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'condition.json').write_text(json.dumps(dict(status='complete',recovery_trials=0,
    comparison_sha256=digest(root/'comparison.json')),indent=2)+'\n')
print(json.dumps(dict(status='complete',true_fatal_layers=[x['layer'] for x in defects],
    camera_prefixes=matched,final=final)),flush=True)
