from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
base=Path('/mnt/data/ishaan/hri-workspace/failure-detection');root=base/'neural-camera-symmetry-v49'
source=base/'neural-nonlinear-refinement-v35/source';sys.path.insert(0,str(source))
from tools.experiment_provenance import runtime_digest,simulation_contract
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
usage=lambda p:int(subprocess.check_output(['du','-sx','-B1',str(p)]).split()[0])
history=base/'neural-trend-historical-v43'
used=usage('/mnt/data/ishaan');reserve=max(0,24_000_000-usage(history)) if not (history/'report-resume.exit').exists() else 0
assert used+reserve+2_000_000<99_990_000_000,(used,reserve)
assert not root.exists()
assert read(base/'neural-functional-camera-gate-v48/comparison.json')['status']=='rejected'
root.mkdir();shutil.copyfile('/tmp/camera-symmetry-v49.py',root/'run.py');shutil.copyfile(__file__,root/'prepare.py')
cases=[]
for profile,part,seed,layers in [('cartesian','mug',1003,[44,45,46,47]),('corexy','mug',1003,[44,45,46,47]),('cartesian','cup',1002,[84,85,86,87]),('bedslinger','cup',1002,[84,85,86,87])]:
    reference=(base/'neural-nonlinear-refinement-v35/cup-under-extrusion-1002/layers.json' if (profile,part)==('cartesian','cup') else base/f'neural-functional-camera-gate-v48/{profile}-{part}-{seed}/layers.json')
    cases.append(dict(name=f'{profile}-{part}-{seed}',profile=profile,part=part,seed=seed,layers=layers,reference_frames=str(reference)))
collection=base/'neural-context-camera-v8/trajectories/trajectories'
groups=[dict(part='cup',trajectory=str(collection/'cartesian-cup-under_extrusion-420.json.gz'),profile_layers={'bedslinger':[87,88]}),
        dict(part='mug',trajectory=str(collection/'cartesian-mug-under_extrusion-420.json.gz'),profile_layers={'cartesian':[90],'corexy':[88,90],'bedslinger':[90]})]
checkpoint=base/'neural-nonlinear-refinement-v35/nonlinear/boundary.npz'
validation=base/'neural-nonlinear-refinement-v35/geometry-validation/frames.jsonl'
pins=[root/'run.py',checkpoint,checkpoint.parent/'training.json',validation,
      source/'tools/inspect_camera_case.py',source/'tools/validate_camera_geometry.py',
      *[Path(c['reference_frames']) for c in cases],*[Path(g['trajectory']) for g in groups]]
p=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='development',
    purpose='Test whether orientation inconsistency causes the remaining camera false alarms, using equal-weight aligned predictions without changing the print specification',
    source_directory=str(source),source_sha256=runtime_digest(source),simulation_contract_sha256=simulation_contract(source)['sha256'],
    checkpoint=str(checkpoint),validation_frames=str(validation),consumed_cases=cases,real_defect_groups=groups,
    candidate='Arithmetic mean of all eight dihedral signed-distance predictions, aligned at the actual native pixel-centre coordinates. Transform current image, actual preceding image and command mask together; no angle or threshold is selected from results.',
    coordinate_gate='All 24 combinations of native scale 1/2/4 and eight symmetries must recover an affine field exactly in the interior; integer-grid complete images must match exactly.',
    diagnostic_gate='All 16 selected false-alarm/adjacent frames must have no immediate fatal geometry; all 12 local support edges must meet the unchanged 6 mm2 limit; all six previously mislabelled real leaks must remain functional defect detections. Ordinary predictions and truth must reproduce retained records.',
    maximum_new_artifact_allowance_bytes=2_000_000,storage_bytes_at_launch=used,other_live_reserved_bytes=reserve,
    reserved_fresh_seeds_untouched=[2001,2002,2003,2004],pins={str(path):digest(path) for path in pins},
    limitation='Selected development diagnostic, not full-frame-grid validation, full-print acceptance or closed-loop recovery. Truth is used only for scoring, never image inference. A pass only qualifies broader camera validation. Existing failures and model weights remain unchanged; no new training or fresh seeds are used.')
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch='#!/bin/bash\nset -uo pipefail\ncd '+str(source)+'\n'
launch+='export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH='+str(source)+' GIT_DIR=/home/ishaan/failure-detection/.git\n'
launch+='/home/ishaan/failure-detection/.venv/bin/python -u '+str(root/'run.py')+' > '+str(root/'run.log')+' 2>&1\n'
launch+='result=$?\nprintf "%s\\n" "$result" > '+str(root/'run.exit')+'\nexit "$result"\n'
(root/'launch.sh').write_text(launch)
window=subprocess.check_output(['tmux','new-window','-d','-t','neural-recovery','-n','camera-symmetry-v49','-P','-F','#{window_id}','bash '+str(root/'launch.sh')],text=True).strip()
(root/'launch.json').write_text(json.dumps(dict(tmux_window=window,session='neural-recovery',protocol_sha256=digest(root/'protocol.json')),indent=2)+'\n')
print(json.dumps(dict(root=str(root),window=window,used=used,reserved=reserve)))
