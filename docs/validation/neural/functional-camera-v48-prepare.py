from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import sys

base = Path('/mnt/data/ishaan/hri-workspace/failure-detection')
root = base/'neural-functional-camera-gate-v48'
source = base/'neural-nonlinear-refinement-v35/source'
sys.path.insert(0, str(source))
from tools.experiment_provenance import runtime_digest, simulation_contract

digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
size = lambda path: int(subprocess.check_output(['du', '-sx', '-B1', str(path)]).split()[0])
used = size('/mnt/data/ishaan')
reserved = max(0, 40_000_000-size(base/'neural-trend-full-v38')) if not (base/'neural-trend-full-v38/run.exit').exists() else 0
history = base/'neural-trend-historical-v43'
if read(history/'condition.json')['status'] == 'running':
    reserved += max(0, 24_000_000-size(history))
assert used+reserved+4_000_000 < 99_990_000_000, (used, reserved)
assert not root.exists()
original = base/'neural-boundary-v1/model/boundary.npz'
refined = base/'neural-nonlinear-refinement-v35/nonlinear/boundary.npz'
known = base/'neural-nonlinear-refinement-v35/cup-under-extrusion-1002/final.json'
geometry = base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl'
assert read(base/'neural-nonlinear-refinement-v35/condition.json')['status'] == 'rejected'
root.mkdir()
shutil.copyfile('/tmp/functional-camera-gate-v48.py', root/'run.py')
shutil.copyfile(__file__, root/'prepare.py')
jobs = []
for profile, part, seed in [('bedslinger', 'cup', 1002), ('cartesian', 'mug', 1003), ('corexy', 'mug', 1003)]:
    name = f'{profile}-{part}-{seed}'
    out = root/name
    command = ['/home/ishaan/failure-detection/.venv/bin/python', str(source/'tools/inspect_camera_case.py'),
        '--profile', profile, '--part', part, '--kind', 'under_extrusion', '--seed', str(seed),
        '--checkpoint', 'original='+str(original), '--checkpoint', 'nonlinear='+str(refined),
        '--include-classical-base-hybrids', '--out', str(out)]
    jobs.append(dict(name=name, case=[profile, part, 'under_extrusion', seed], out=str(out), command=command))
p = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage='development',
    purpose='Test a new supplemental-camera hypothesis against unchanged functional defect criteria; retain the earlier per-feature rejection',
    source_directory=str(source), source_sha256=runtime_digest(source),
    simulation_contract_sha256=simulation_contract(source)['sha256'],
    geometry_frames=str(geometry), known_cup_verdict=str(known), jobs=jobs,
    maximum_new_artifact_allowance_bytes=4_000_000, storage_bytes_at_launch=used,
    other_live_job_reserved_bytes=reserved, reserved_fresh_seeds_untouched=[2001,2002,2003,2004],
    geometry_gate='Every original true per-layer fatal defect detection must remain detected by the unchanged specification, and aggregate fatal misses may not increase. Extra missed leak labels must still be hard-thin defects. False positives and all category counts remain reported. Specified bases use unchanged classical observations.',
    diagnostic_gate='All three new consumed nominal-command prints plus the retained cup must actually pass the unchanged full functional check under the supplemental measurement; actual ignore successes must reproduce.',
    prospective_integration='Only if these gates pass, consider an additional observed camera history when every existing executable candidate fails. Existing passing decisions must remain unchanged. Closed-loop Qwen, defect, preservation, historical and untouched comparisons remain mandatory.',
    limitation='Selected retrospective development diagnostics; no recovery or AI-agent result is fabricated. The v35 per-feature gate remains failed. A different defect label can still reject an unsafe print under the unchanged functional specification; this hypothesis needs complete closed-loop testing. No reserved fresh seeds are used.',
    pins={str(path):digest(path) for path in [root/'run.py', original, refined, known, geometry,
        base/'neural-nonlinear-refinement-v35/nonlinear/training.json',
        base/'neural-nonlinear-refinement-v35/protocol.json',
        base/'neural-nonlinear-refinement-v35/geometry-validation/summary.json',
        source/'tools/inspect_camera_case.py']})
(root/'protocol.json').write_text(json.dumps(p, indent=2)+'\n')
launch = '#!/bin/bash\nset -uo pipefail\ncd '+str(source)+'\n'
launch += 'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH='+str(source)+' GIT_DIR=/home/ishaan/failure-detection/.git\n'
launch += '/home/ishaan/failure-detection/.venv/bin/python -u '+str(root/'run.py')+' > '+str(root/'run.log')+' 2>&1\n'
launch += 'result=$?\nprintf "%s\\n" "$result" > '+str(root/'run.exit')+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window = subprocess.check_output(['tmux', 'new-window', '-d', '-t', 'neural-recovery', '-n', 'functional-camera-v48', '-P', '-F', '#{window_id}', 'bash '+str(root/'launch.sh')], text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window, session='neural-recovery', protocol_sha256=digest(root/'protocol.json')), indent=2)+'\n')
print(json.dumps(dict(root=str(root), tmux_window=window, used=used, reserved=reserved)))
