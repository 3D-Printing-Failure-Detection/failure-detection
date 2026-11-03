from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

repo = Path('/home/ishaan/failure-detection')
base = Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root, source = base / 'neural-functional-band-v66', base / 'neural-functional-band-v66/source'
read = lambda path: json.loads(Path(path).read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
size = lambda path: int(subprocess.check_output(['du', '-sx', '-B1', str(path)]).split()[0])
used = size('/mnt/data/ishaan')
other_root = base / 'neural-observed-wall-v62'
reserved = max(0, 18_000_000-size(other_root)) + 1_000_000
assert used + 4_000_000 + reserved < 99_990_000_000, (used, reserved)
assert (base / 'neural-guided-refinement-v63/run.exit').read_text().strip() == '0'
assert read(base / 'neural-guided-refinement-v63/condition.json')['status'] == 'rejected'
assert (base / 'neural-functional-band-v64/run.exit').read_text().strip() == '1'
assert read(base / 'neural-functional-band-v64/condition.json')['error'] == "AssertionError(('target replay mismatch', 1))"
assert (base / 'neural-functional-band-v65/run.exit').read_text().strip() == '1'
assert 'TopologyException' in read(base / 'neural-functional-band-v65/condition.json')['error']
root.mkdir()
source.mkdir()
for pattern in ['recovery/**/*.py', 'tools/*.py', 'parts/*.yaml']:
    for path in repo.glob(pattern):
        destination = source / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract
shutil.copy2('/tmp/functional-band-v66-run.py', root / 'run.py')
shutil.copy2(__file__, root / 'prepare.py')
original = base / 'neural-command-camera-v15/dataset/data.npz'
extra = base / 'neural-training-coverage-v56/dataset/data.npz'
labels = base / 'neural-coverage-weighting-v57/dataset/critical-material.npz'
trajectories = base / 'neural-context-camera-v8/trajectories/trajectories'
groups = {row['group'] for row in read(original.parent / 'manifest.json')['groups']}
groups |= {row['parent_group'] for row in read(extra.parent / 'manifest.json')['groups']}
pins = [root / 'run.py', root / 'prepare.py', original, original.parent / 'manifest.json', extra,
        extra.parent / 'manifest.json', labels, labels.parent / 'manifest.json',
        source / 'tools/boundary_functional_band.py', source / 'tools/boundary_training_augmentation.py',
        source / 'tools/boundary_tiles.py', source / 'tools/experiment_provenance.py',
        source / 'tools/add_boundary_command_input.py', base / 'neural-functional-band-v64/condition.json',
        base / 'neural-functional-band-v65/condition.json', base / 'neural-functional-band-v65/run.py',
        base / 'neural-functional-band-v65/protocol.json',
        base / 'neural-functional-band-v64/run.py', base / 'neural-functional-band-v64/protocol.json',
        base / 'neural-guided-refinement-v63/screen.json', base / 'neural-guided-refinement-v63/condition.json']
pins += [trajectories / (group + '.json.gz') for group in groups]
p = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage='supervision-audit',
         source=str(source), source_sha256=runtime_digest(source),
         simulation_contract_sha256=simulation_contract(source)['sha256'],
         original_data=str(original), extra_data=str(extra), critical_labels=str(labels), trajectories=str(trajectories),
         pins={str(path):digest(path) for path in pins}, maximum_new_artifact_allowance_bytes=4_000_000,
         storage_bytes_at_launch=used, other_live_reserved_bytes=reserved,
         concurrent_reservation=dict(root=str(other_root), maximum_artifact_bytes=18_000_000, supervisor_log_bytes=1_000_000),
         derived_geometry_repair='Only invalid morphological openings are made valid. Absolute area change must be at most1e-9 * max(1, opening area); otherwise fail. Original geometry, existing targets, physical scoring and old labels are unchanged. v65 failed at original row5291 on a valid deposit whose derived opening self-intersected; repairing that opening changes area by-2.2737367544323206e-13 mm2.',
         target_replay='Original tiles use float64 -> float32 -> float16 exactly as add_boundary_command_input.py; augmented tiles use float64 -> float16 exactly as boundary_training_augmentation.py. No stored target is changed.',
         predecessor='v64 stopped at original row1 because one pixel differs by0.000244140625 under direct half-precision rounding. Its frozen failure is retained. v66 restores the original documented intermediate float32 cast without weakening exact equality.',
         purpose='Audit boundary supervision near the actual hard-wall limit after two missed cases prove to be closed but excessively thin walls.',
         definition='Use the unchanged specification opening at HARD_WALL_FRAC * min_wall_mm / 2, with quad_segs4. Subtract nominal thin material, buffer the additional thin region by0.15mm, and rasterize at existing native query coordinates. Union all original critical-positive labels. This labels both sides of the thin boundary to supervise width, not only the survival of positive material.',
         acceptance='Reproduce all7104 signed-distance targets and every augmented geometry hash; preserve all original labels, arrays, group ordering and train/validation split. Record positive, negative and zero-target label counts separately. Validation labels are audit-only and may never set training normalization or fitted weights.',
         reserved_fresh_seeds_untouched=[2001, 2002, 2003, 2004],
         limitation='Supervision audit only. No model training, recovery run or qualification follows automatically. A wider supervision band is not a change to camera inputs or physical scoring.')
assert p['source_sha256'] == runtime_digest(repo)
assert p['simulation_contract_sha256'] == 'fb46bdd5177b2582838ca303474f2426d40ea0396cedfc27097e3ba115eb95b6'
(root / 'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
launch = '#!/bin/bash\nset -uo pipefail\ncd ' + shlex.quote(str(source)) + '\n'
launch += 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl\n'
launch += shlex.join([str(repo / '.venv/bin/python'), '-u', str(root / 'run.py')]) + ' > ' + shlex.quote(str(root / 'run.log')) + ' 2>&1\n'
launch += 'result=$?\nprintf "%s\\n" "$result" > ' + shlex.quote(str(root / 'run.exit')) + '\nexit "$result"\n'
(root / 'launch.sh').write_text(launch)
window = subprocess.check_output(['tmux', 'new-window', '-d', '-t', 'neural-recovery', '-n', 'functional-band-v66',
                                  '-P', '-F', '#{window_id}', 'bash ' + shlex.quote(str(root / 'launch.sh'))], text=True).strip()
(root / 'launch.json').write_text(json.dumps(dict(tmux_window=window, session='neural-recovery',
                                                protocol_sha256=digest(root / 'protocol.json'), runner_sha256=digest(root / 'run.py')),
                                           indent=2)+'\n')
print(json.dumps(dict(root=str(root), window=window, used=used, concurrent_reservation=reserved)))
