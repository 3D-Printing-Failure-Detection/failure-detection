"""Start the pinned recovery trial only after the existing camera run qualifies."""
from pathlib import Path
import hashlib,json,subprocess,time

root=Path(__file__).resolve().parent
p=json.loads((root/'protocol.json').read_text())
camera=Path(p['camera_root'])
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(status,**extra):
    value=dict(status=status,**extra)
    (root/'condition.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

def live_identity():
    proc=Path('/proc')/str(p['camera_pid'])
    try:
        command=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        # Fields after the final ')' begin with state (field 3); starttime is 22.
        start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        return command==p['camera_command'] and start==p['camera_starttime']
    except FileNotFoundError:
        return False

save('waiting_for_camera_gate',camera_pid=p['camera_pid'])
while not (camera/'run.exit').exists():
    if not live_identity():
        # Give the parent launch shell time to write its terminal exit record.
        time.sleep(2)
        if not (camera/'run.exit').exists():
            save('stopped_without_exit_record');raise SystemExit(1)
    time.sleep(15)
for path,expected in p['pins'].items():assert digest(path)==expected,path
exit_code=int((camera/'run.exit').read_text())
gate=json.loads((camera/'condition.json').read_text())
if exit_code!=0 or gate['status']!='camera_gates_passed':
    save('skipped',camera_exit=exit_code,camera_status=gate['status'])
    raise SystemExit(0)
save('launching_qualified_recovery',geometry_gate=gate['geometry_gate'])
result=subprocess.run(p['command'],cwd=p['repository'])
save('launched' if result.returncode==0 else 'launch_failed',launch_exit=result.returncode)
raise SystemExit(result.returncode)
