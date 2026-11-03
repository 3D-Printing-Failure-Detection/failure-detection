"""Pack terminal audited results, then verify an actual disk restoration."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,shutil,subprocess,sys

repo=Path('/home/ishaan/failure-detection');base=Path('/mnt/data/ishaan/hri-workspace/failure-detection')
sys.path.insert(0,str(repo))
from tools.result_storage import pack_runs,load_run_bytes,verifier_digest

read=lambda path:json.loads(Path(path).read_text())
digest=lambda value:hashlib.sha256(value).hexdigest()
size=lambda path:int(subprocess.check_output(['du','-sx','-B1',str(path)]).split()[0])
root=base/'completed-primary-packing-v13'
targets=[base/name for name in ['neural-robust-full-v33','neural-robust-historical-v34','neural-trend-historical-v43']]
camera=base/'neural-native-coverage-v58'
protocol=read(camera/'protocol.json')
reserved=max(0,protocol['maximum_new_artifact_allowance_bytes']-size(camera))
used=size('/mnt/data/ishaan');assert used+reserved+2_000_000<99_990_000_000
directories=[]
for target in targets:
    assert (target/'run.exit').read_text().strip()=='0'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert str(target/'run.py').encode() not in command,'target runner is still live'
    directories += sorted(path.parent for path in target.glob('**/runs.jsonl') if 'source' not in path.parts)
assert len(directories)==13
root.mkdir();shutil.copy2(__file__,root/'run.py')
records=[]
for directory in directories:
    raw=load_run_bytes(directory);manifest=(directory/'manifest.json').read_bytes()
    audit=(directory.parent/'audit/audit.json').read_bytes()
    assert read(directory/'manifest.json')['status']=='complete'
    assert read(directory.parent/'audit/audit.json')['status']=='passed'
    # Compression and verification need only the compressed bytes above the
    # original footprint; raw is unlinked after the packer's stored round-trip.
    compressed_size=len(gzip.compress(raw,compresslevel=6,mtime=0))
    assert size('/mnt/data/ishaan')+reserved+compressed_size+200_000<99_990_000_000
    before=size(directory)
    receipt=pack_runs(directory)
    restored=root/'restored-runs.jsonl'
    with gzip.open(directory/'runs.jsonl.gz','rb') as source,restored.open('xb') as output:
        shutil.copyfileobj(source,output)
    restored_bytes=restored.read_bytes()
    assert restored_bytes==raw and digest(restored_bytes)==receipt['original_sha256']
    assert load_run_bytes(directory)==raw
    assert (directory/'manifest.json').read_bytes()==manifest
    assert (directory.parent/'audit/audit.json').read_bytes()==audit
    restored.unlink()
    after=size(directory)
    records.append(dict(directory=str(directory),rows=len(raw.splitlines()),original_bytes=len(raw),
        allocated_bytes_reclaimed=before-after,receipt_sha256=digest((directory/'runs-storage.json').read_bytes()),
        actual_disk_restore_verified=True,original_sha256=receipt['original_sha256']))
    (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
result=dict(status='verified',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    script_sha256=digest((root/'run.py').read_bytes()),packer_sha256=verifier_digest(),
    directories=len(records),rows=sum(row['rows'] for row in records),
    allocated_bytes_reclaimed=sum(row['allocated_bytes_reclaimed'] for row in records),records=records,
    storage_before_bytes=used,storage_after_bytes=size('/mnt/data/ishaan'),
    unchanged='All decoded primary result bytes, manifests and audits. No active runs, model weights or training data changed.',
    readers='Use tools.result_storage.load_run_bytes. Legacy scripts that directly open runs.jsonl require restoration first; their original files and manifests are retained unchanged.',
    restore='Decompress each runs.jsonl.gz to runs.jsonl, verify original_sha256 in runs-storage.json, then remove the gzip copy; the reader rejects both formats being present simultaneously.')
(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
