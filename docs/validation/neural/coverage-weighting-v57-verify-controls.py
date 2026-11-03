"""Compare every completed control layer without changing the frozen runner."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
read = lambda path: json.loads(path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
protocol = read(root/'protocol.json')
reference = Path(protocol['coverage_root'])
records = []
for job in protocol['diagnostics']:
    current, prior = root/job['name'], reference/job['name']
    if not (current/'manifest.json').exists() or read(current/'manifest.json')['status'] != 'complete':
        continue
    assert read(prior/'manifest.json')['status'] == 'complete'
    for directory in (current, prior):
        manifest = read(directory/'manifest.json')
        assert manifest['source_sha256'] == manifest['final_source_sha256'] == protocol['source_sha256']
        assert manifest['frames'] == manifest['layers_printed'] == 160
        assert manifest['follows_nominal_commands'] and manifest['recovery_actions'] == 0
    layers, old = read(current/'layers.json'), read(prior/'layers.json')
    assert len(layers) == len(old) == 160
    for now, before in zip(layers, old):
        assert all(now[key] == before[key] for key in ('layer', 'z', 'truth'))
        for name, previous in [('classical', 'classical'), ('control', 'coverage'),
                               ('control:classical-base', 'coverage:classical-base')]:
            assert now['measured'][name] == before['measured'][previous], (job['name'], now['layer'], name)
    final, earlier = read(current/'final.json'), read(prior/'final.json')
    assert final['truth'] == earlier['truth']
    for name, previous in [('classical', 'classical'), ('control', 'coverage'),
                           ('control:classical-base', 'coverage:classical-base')]:
        assert final['measured'][name] == earlier['measured'][previous]
    records.append(dict(case=job['case'], frames=160, current=str(current), reference=str(prior),
                        current_layers_sha256=digest(current/'layers.json'), reference_layers_sha256=digest(prior/'layers.json'),
                        current_final_sha256=digest(current/'final.json'), reference_final_sha256=digest(prior/'final.json')))
assert records, 'no completed diagnostic to verify'
frames = sum(row['frames'] for row in records)
result = dict(status='verified', complete=len(records) == len(protocol['diagnostics']), frames=frames,
              script_sha256=digest(Path(__file__)), protocol_sha256=digest(root/'protocol.json'), records=records,
              scope='Exact per-layer truth, camera features, IoU, boundary distance, and final verdict/metrics',
              limitation='These diagnostics retain no per-frame RGB hashes or complete measured geometry; neither is claimed to match byte-for-byte.')
output = root/f'controls-{frames}.json'
assert not output.exists(), 'preserve earlier verification records'
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(status='verified', complete=result['complete'], frames=frames, output=str(output))))
