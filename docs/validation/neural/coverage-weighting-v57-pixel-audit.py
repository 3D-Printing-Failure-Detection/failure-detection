from pathlib import Path
import hashlib,json,sys
import numpy as np

root=Path('/mnt/data/ishaan/hri-workspace/failure-detection/neural-coverage-weighting-v57')
base=root.parent;source=root/'source';sys.path.insert(0,str(source))
from recovery.models.boundary import BoundarySegmenter
from tools.experiment_provenance import runtime_digest
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
reference=read(base/'neural-critical-pixel-camera-v55/critical-pixel-audit-verified.json')
dataset=base/'neural-command-camera-v15/dataset/data.npz'
labels=base/'neural-critical-pixel-camera-v55/dataset/critical-material.npz'
assert runtime_digest(source)==reference['source_sha256']
assert digest(dataset)==reference['dataset_sha256'] and digest(labels)==reference['mask_sha256']
with np.load(dataset,allow_pickle=False) as data:
    x,y,split,groups=[data[k] for k in ('x','y','validation','group')]
with np.load(labels,allow_pickle=False) as data:
    masks=data['critical_material'].astype(bool)
    assert np.array_equal(data['group'],groups) and np.array_equal(data['validation'],split)
models=dict(control=base/'neural-command-camera-v15/command/boundary.npz',weighted=base/'neural-critical-pixel-camera-v55/candidate/boundary.npz',coverage=base/'neural-training-coverage-v56/candidate/boundary.npz',combined=root/'candidate/boundary.npz')
summary={name:{} for name in ('training','validation')}
for name,path in models.items():
    model=BoundarySegmenter(path)
    assert model.sha256==read(path.parent/'training.json')['checkpoint_sha256']
    for partition,held_out in [('training',False),('validation',True)]:
        counts=dict(critical_positive_pixels=0,critical_correct=0,other_positive_pixels=0,
                    other_positive_correct=0,negative_pixels=0,false_positive_pixels=0)
        for i in np.flatnonzero((split==held_out)&masks.any(axis=(1,2,3))):
            patches=np.lib.stride_tricks.sliding_window_view(x[i].astype('float32'),(9,9),axis=(-2,-1))
            patches=patches.transpose(1,2,0,3,4).reshape(-1,243)
            pred=model.predict_patches(patches).reshape(16,16,4,4).transpose(0,2,1,3).reshape(64,64)>0
            rare=masks[i,0];other=(y[i,0]>0)&~rare;negative=y[i,0]<=0
            values=[rare.sum(),(pred&rare).sum(),other.sum(),(pred&other).sum(),negative.sum(),(pred&negative).sum()]
            for key,value in zip(counts,values):counts[key]+=int(value)
        counts['critical_recall']=counts['critical_correct']/counts['critical_positive_pixels']
        summary[partition][name]=counts
        if name in ('control','weighted'):
            assert counts==reference['summary'][partition]['control' if name=='control' else 'candidate']
assert all(summary[part]['coverage']==read(base/'neural-training-coverage-v56/critical-pixel-audit.json')['summary'][part]['coverage'] for part in summary)
result=dict(status='complete',stage='retrospective_fixed_checkpoint_audit_not_selection',
            source_sha256=runtime_digest(source),script_sha256=digest(__file__),
            dataset_sha256=digest(dataset),mask_sha256=digest(labels),
            reference_sha256=digest(base/'neural-critical-pixel-camera-v55/critical-pixel-audit-verified.json'),
            checkpoints={name:dict(path=str(path),sha256=digest(path)) for name,path in models.items()},
            summary=summary,old_control_and_weighted_counts_reproduced_exactly=True,
            limitation='Only the original 65 rare-material tiles; correlated pixel counts do not establish geometric or recovery success. No epoch or model is selected here.')
out=root/'critical-pixel-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
