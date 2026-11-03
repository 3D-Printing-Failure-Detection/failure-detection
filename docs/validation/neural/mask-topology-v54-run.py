from pathlib import Path
from dataclasses import replace
import hashlib
import json
import sys
import time

root=Path(__file__).resolve().parent
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');sys.path.insert(0,p['source'])
import cv2
import numpy as np
import shapely
from recovery import camera
from recovery.eval.camera_trace import camera_frame_digest
from recovery.geometry.ops import clean,polys
from recovery.models.boundary import BoundarySegmenter
from recovery.parts import get_part
from recovery.types import Layer
from tools.inspect_camera_case import layer_features
from tools.experiment_provenance import runtime_digest


def pixel_cells(mask,cfg):
    # Exact union of the observed foreground pixel squares: no closing,
    # buffering, hole filling, or additional positive pixel is allowed.
    edges=np.diff(np.pad(mask.astype('int8'),((0,0),(1,1))),axis=1)
    yy,start=np.nonzero(edges==1);ys,stop=np.nonzero(edges==-1)
    assert np.array_equal(yy,ys)
    if not len(start):return shapely.Polygon()
    x0=(start-.5-cfg.size/2)/cfg.px_per_mm;x1=(stop-.5-cfg.size/2)/cfg.px_per_mm
    y0=(cfg.size/2-yy-.5)/cfg.px_per_mm;y1=(cfg.size/2-yy+.5)/cfg.px_per_mm
    united=shapely.union_all(shapely.box(x0,y0,x1,y1))
    assert abs(united.area-mask.sum()/cfg.px_per_mm**2)<1e-7
    return clean(shapely.union_all([part for part in polys(united) if part.area>=4.])).simplify(.03)


assert runtime_digest(p['source'])==p['source_sha256']
for path,sha in p['pins'].items():assert digest(path)==sha,path
refs={(r['case'],r['layer']):r for r in map(json.loads,Path(p['reference_frames']).read_text().splitlines())}
rows=[]
for case in p['cases']:
    image=cv2.imread(case['image']);previous=cv2.imread(case['previous'])
    ref=refs[case['name'],case['layer']]
    assert camera_frame_digest(image).hex()==ref['image_sha256_128']
    assert camera_frame_digest(previous).hex()==ref['previous_image_sha256_128']
    part=get_part(case['part']);nominal=part.layers();index=case['layer']
    command=nominal[index];mask=camera.expected_mask(command);prior=camera.expected_mask(nominal[index-1])
    for entry in p['models']:
        started=time.perf_counter();model=BoundarySegmenter(entry['checkpoint']);model.prime_history(previous,prior)
        distance=model.signed_distance(image,mask)
        if model.upsample>model.output_upsample:
            factor=model.upsample//model.output_upsample
            xx,yy=np.meshgrid(np.arange(distance.shape[1]*factor,dtype='float32')/factor,
                              np.arange(distance.shape[0]*factor,dtype='float32')/factor)
            distance=cv2.remap(distance,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        binary=distance>0;cfg=replace(camera.CAM,px_per_mm=camera.CAM.px_per_mm*model.upsample)
        contours,hierarchy=cv2.findContours(binary.astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        holes=([cv2.contourArea(c)/cfg.px_per_mm**2 for c,h in zip(contours,hierarchy[0]) if h[3]>=0]
               if hierarchy is not None else [])
        old=clean(model.measure(image,mask).simplify(.25/4));old_features=layer_features(Layer(command.z,old),command,part.spec)
        for key,value in ref['measured'][entry['name']].items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):assert abs(old_features[key]-value)<1e-7
            else:assert old_features[key]==value
        cell=clean(pixel_cells(binary,cfg).simplify(.25/4))
        features=layer_features(Layer(command.z,cell),command,part.spec)
        row=dict(case=case['name'],layer=index,model=entry['name'],checkpoint_sha256=model.sha256,
            largest_raw_mask_hole_mm2=max(holes,default=0.),old=old_features,pixel_cells=features,
            foreground_pixels=int(binary.sum()),mask_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),
            seconds=time.perf_counter()-started)
        rows.append(row)
        print(case['name'],entry['name'],'raw hole',row['largest_raw_mask_hole_mm2'],
              'old gap',old_features.get('open_wall'),'cell gap',features.get('open_wall'),flush=True)
        (root/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
result=dict(status='complete',rows=len(rows),original_features_reproduced=True,
    unchanged_input_images=True,protocol_sha256=digest(root/'protocol.json'),rows_sha256=digest(root/'rows.json'),
    cases={case['name']:[r['model'] for r in rows if r['case']==case['name'] and not any(r['pixel_cells'].get(k,False)
        for k in ('open_wall','hard_thin','base_problem','connection_problem'))] for case in p['cases']},
    limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
