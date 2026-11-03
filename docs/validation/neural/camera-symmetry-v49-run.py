from pathlib import Path
import gzip, hashlib, json, sys

root=Path(__file__).resolve().parent
p=json.loads((root/'protocol.json').read_text());source=Path(p['source_directory'])
sys.path.insert(0,str(source))
import cv2
import numpy as np
import shapely
from recovery import camera
from recovery.deform import random_failure
from recovery.geometry import features as F
from recovery.geometry.ops import clean
from recovery.geometry.spec import OVERHANG_AREA
from recovery.models.boundary import BoundarySegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.inspect_camera_case import layer_features
from tools.validate_camera_geometry import frame_indices
from tools.experiment_provenance import runtime_digest,simulation_contract

read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
image_digest=lambda image:hashlib.sha256(image.dtype.str.encode()+str(image.shape).encode()+image.tobytes()).hexdigest()
flags=['open_wall','hard_thin','base_problem','connection_problem']
fatal=lambda value:any(value.get(k,False) for k in flags)


def verify():
    assert runtime_digest(source)==p['source_sha256']
    assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
    for path,expected in p['pins'].items():assert digest(path)==expected,path


def transform(value,turns,mirror):
    rotated=np.rot90(value,turns)
    return np.ascontiguousarray(np.fliplr(rotated) if mirror else rotated)


def aligned(distance,shape,scale,turns,mirror):
    # Native samples sit at integer pixels plus offsets 0,1/s,...,(s-1)/s.
    # Rotating the native array itself would incorrectly shift it by s-1 samples.
    height,width=shape
    y,x=np.mgrid[:height*scale,:width*scale].astype('float32')/scale
    if turns==0:u,v=x,y
    elif turns==1:u,v=y,width-1-x
    elif turns==2:u,v=width-1-x,height-1-y
    elif turns==3:u,v=height-1-y,x
    else:raise ValueError(turns)
    rotated_width=height if turns%2 else width
    if mirror:u=rotated_width-1-u
    u,v=u*scale,v*scale
    valid=(u>=0)&(v>=0)&(u<=distance.shape[1]-1)&(v<=distance.shape[0]-1)
    result=cv2.remap(distance,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return result,valid


def coordinate_check():
    records=[]
    for scale in [1,2,4]:
        h,w=9,11;y,x=np.mgrid[:h,:w].astype('float32');coarse=x+3*y
        yy,xx=np.mgrid[:h*scale,:w*scale].astype('float32')/scale
        expected=xx+3*yy
        for turns in range(4):
            for mirror in [False,True]:
                moved=transform(coarse,turns,mirror)
                sy,sx=np.mgrid[:moved.shape[0]*scale,:moved.shape[1]*scale].astype('float32')/scale
                native=cv2.remap(moved,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                restored,valid=aligned(native,(h,w),scale,turns,mirror)
                interior=valid&(xx>=1)&(xx<=w-2)&(yy>=1)&(yy<=h-2)
                error=float(np.max(np.abs(restored[interior]-expected[interior])))
                assert error<1e-6,(scale,turns,mirror,error)
                if scale==1:assert np.array_equal(restored,coarse) and valid.all()
                records.append(dict(scale=scale,turns=turns,mirror=mirror,max_interior_error=error))
    return records


def region_from_distance(model,distance):
    if model.upsample>model.output_upsample:
        factor=model.upsample//model.output_upsample
        y,x=np.mgrid[:distance.shape[0]*factor,:distance.shape[1]*factor].astype('float32')/factor
        distance=cv2.remap(distance,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    cfg=camera.C.CameraCfg(px_per_mm=camera.CAM.px_per_mm*model.upsample,fov_mm=camera.CAM.fov_mm)
    return clean(camera.mask_to_region((distance>0).astype('uint8'),cfg,min_area=4.,simplify=.03).simplify(.25/4))


def inspect_frame(model,image,previous,command,actual,part,reference):
    expected=camera.expected_mask(command)
    total=None;counts=None;individual=[];ordinary=None
    for turns in range(4):
        for mirror in [False,True]:
            model.prime_history(transform(previous,turns,mirror))
            distance=model.signed_distance(transform(image,turns,mirror),transform(expected,turns,mirror))
            restored,valid=aligned(distance,image.shape[:2],model.output_upsample,turns,mirror)
            if total is None:total=np.zeros_like(restored,dtype='float64');counts=np.zeros_like(restored,dtype='uint8')
            total+=np.where(valid,restored,0);counts+=valid
            layer=Layer(command.z,region_from_distance(model,restored))
            individual.append(dict(turns=turns,mirror=mirror,features=layer_features(layer,command,part.spec)))
            if turns==0 and not mirror:ordinary=layer
    assert counts.min()>=1
    averaged=Layer(command.z,region_from_distance(model,(total/counts).astype(model.inference_dtype)))
    baseline=layer_features(ordinary,command,part.spec)
    # Preserve the ordinary prediction and offline truth of the earlier replay.
    for key,value in reference['measured']['nonlinear'].items():
        if key not in baseline:continue
        if isinstance(value,(int,float)) and not isinstance(value,bool):assert abs(baseline[key]-value)<1e-7,(key,baseline[key],value)
        else:assert baseline[key]==value,(key,baseline[key],value)
    truth=layer_features(actual,command,part.spec)
    for key,value in reference['truth'].items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):assert abs(truth[key]-value)<1e-7
        else:assert truth[key]==value
    return dict(image_sha256=image_digest(image),previous_image_sha256=image_digest(previous),
        truth=truth,ordinary=baseline,mean_eight=layer_features(averaged,command,part.spec),
        orientations=individual,all_ordinary_reference_features_reproduced=True,
        min_orientation_count=int(counts.min()),interior_orientation_count=int(counts[4:-4,4:-4].min())),ordinary,averaged


verify()
coordinates=coordinate_check();(root/'coordinate-check.json').write_text(json.dumps(coordinates,indent=2)+'\n')
model=BoundarySegmenter(p['checkpoint']);rows=[];pair_results=[]
for case in p['consumed_cases']:
    print('START consumed',case['name'],flush=True)
    part=get_part(case['part']);nominal=part.layers()
    reference={r['layer']:r for r in read(case['reference_frames'])}
    failure=random_failure(nominal,np.random.default_rng(case['seed']),kind='under_extrusion')
    selected=set(case['layers']);histories={name:{} for name in ['actual','ordinary','mean_eight']}
    with MujocoPrinter(nominal,failure,case['seed'],case['profile'],False) as printer:
        previous=None
        for index in range(max(selected)+1):
            actual=printer.print_layer();image=printer.camera_image()
            if index in selected:
                row,ordinary,averaged=inspect_frame(model,image,image if previous is None else previous,
                    nominal[index],actual,part,reference[index])
                row.update(case=case['name'],layer=index,split='consumed_false_alarm')
                rows.append(row)
                for name,layer in [('actual',actual),('ordinary',ordinary),('mean_eight',averaged)]:histories[name][index]=layer
                if case['name']=='cartesian-mug-1003' and index==45:
                    assert cv2.imwrite(str(root/'mug-layer45.png'),image)
                print('FRAME',case['name'],index,{name:row[name] for name in ['truth','ordinary','mean_eight']},flush=True)
            previous=image
    pairs=[]
    for index in sorted(selected):
        if index-1 not in selected:continue
        pairs.append(dict(layer=index,excess_overhang_mm2={name:F.excess_overhang(history[index],history[index-1],nominal[index],nominal[index-1]) for name,history in histories.items()}))
    pair_results.append(dict(case=case['name'],pairs=pairs))
    (root/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print('PAIRS',json.dumps(pair_results[-1]),flush=True)

validation=[json.loads(line) for line in Path(p['validation_frames']).read_text().splitlines()]
for job in p['real_defect_groups']:
    with gzip.open(job['trajectory'],'rt') as stream:record=json.load(stream)
    assert record['validation']
    part=get_part(job['part']);nominal=part.layers()
    actual=[Layer(frame['z'],shapely.from_wkb(frame['actual'])) for frame in record['frames']]
    indices=frame_indices(record,part,12)
    for profile,selected in job['profile_layers'].items():
        print('START real defects',record['group'],profile,flush=True)
        previous_seed=int.from_bytes(hashlib.sha256((record['group']+profile+'-previous-camera').encode()).digest()[:8],'little')
        with MujocoPrinter(nominal,None,420,profile,False) as printer,MujocoPrinter(nominal,None,previous_seed,profile,False) as previous_printer:
            for index in indices:
                printer.printed=actual[:index+1];image=printer.camera_image();previous=image
                if index>0:
                    previous_printer.printed=actual[:index];previous=previous_printer.camera_image()
                if index not in selected:continue
                ref=next(r for r in validation if (r['group'],r['profile'],r['layer'])==(record['group'],profile,index))
                row,_,_=inspect_frame(model,image,previous,nominal[index],actual[index],part,ref)
                row.update(case=record['group']+':'+profile,layer=index,split='selected_real_defect')
                assert fatal(row['truth']) and fatal(row['ordinary'])
                rows.append(row)
                print('REAL FRAME',row['case'],index,'ordinary_fatal',fatal(row['ordinary']),'mean_fatal',fatal(row['mean_eight']),flush=True)
        (root/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
verify()
consumed=[r for r in rows if r['split']=='consumed_false_alarm']
real=[r for r in rows if r['split']=='selected_real_defect']
assert len(consumed)==16 and len(real)==6
no_false_geometry=all(not fatal(r['mean_eight']) for r in consumed)
support_passes=all(pair['excess_overhang_mm2']['mean_eight']<=OVERHANG_AREA for result in pair_results for pair in result['pairs'])
real_preserved=all(fatal(r['mean_eight']) for r in real)
summary=dict(status='passed' if no_false_geometry and support_passes and real_preserved else 'rejected',
    frames=len(rows),consumed_frames=len(consumed),selected_real_defect_frames=len(real),
    all_ordinary_reference_features_reproduced=True,coordinate_checks=24,
    no_false_fatal_geometry=no_false_geometry,local_support_passes=support_passes,
    selected_true_defect_detections_preserved=real_preserved,local_support=pair_results,
    false_fatal_frames=[dict(case=r['case'],layer=r['layer'],features=r['mean_eight']) for r in consumed if fatal(r['mean_eight'])],
    lost_true_detections=[dict(case=r['case'],layer=r['layer']) for r in real if not fatal(r['mean_eight'])],
    protocol_sha256=digest(root/'protocol.json'),frames_sha256=digest(root/'frames.jsonl'),
    new_recovery_trials=0,limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
