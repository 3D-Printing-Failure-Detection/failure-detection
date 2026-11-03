from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(root/'receipt.json')
assert digest(root/'tools/compact_result_exports.py')==r['tool_sha256']
assert digest(root/'tools/result_storage.py')==r['storage_helper_sha256']
sys.path.insert(0,str(root))
from tools.compact_result_exports import restore_exports
choices={x['directory']:x for x in r['inputs']}
assert sys.argv[1:],'Provide one or more result directories listed in receipt.json'
for value in sys.argv[1:]:
    directory=str(Path(value).resolve());entry=choices[directory]
    assert digest(Path(directory)/'exports-storage.json')==entry['receipt_sha256']
    for path,expected in entry['pins'].items():assert digest(path)==expected
    used=int(subprocess.check_output(['du','-sx','-B1','/mnt/data/ishaan']).split()[0])
    assert used+entry['original_bytes']+8192*entry['files']<100_000_000_000,'restore exceeds quota'
    print(json.dumps(restore_exports(directory)))
