from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parent
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');source=Path(p['source']);sys.path.insert(0,str(source))
import cv2
import numpy as np
import mujoco
from recovery import camera
from recovery.deform import random_failure
from recovery.eval.camera_trace import camera_frame_digest
from recovery.geometry import features as F
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.types import Layer
from tools.experiment_provenance import runtime_digest,simulation_contract


def verify():
    assert runtime_digest(source)==p['source_sha256']
    assert simulation_contract(source)['sha256']==p['simulation_contract_sha256']
    for path,sha in p['pins'].items():assert digest(path)==sha,path


def visibility(printer,current_start):
    scene=printer.renderer.scene
    for index in range(scene.ngeom):
        geom=scene.geoms[index]
        geom.segid=1 if index>=current_start else 0
        geom.objid=1 if index>=current_start else 0
        geom.objtype=int(mujoco.mjtObj.mjOBJ_GEOM)
    printer.renderer.enable_segmentation_rendering()
    try:
        labels=printer.renderer.render()
    finally:
        printer.renderer.disable_segmentation_rendering()
    mask=(labels[:,:,0]==1).astype('uint8')*255
    return printer._calibration.top_down(mask,printer.printed[-1].z,(0,0))


def enclosed(mask,command):
    region=camera.mask_to_region((mask>127).astype('uint8'),min_area=4.,simplify=.03)
    actual=F.base(Layer(command.z,region));expected=F.base(command)
    return dict(hole_area_mm2=actual['hole_area'],area_mm2=actual['area'],
                open_wall=actual['hole_area']<.5*expected['hole_area'])


verify();rows=[]
reference={ (row['case'],row['layer']):row for row in map(json.loads,Path(p['reference_frames']).read_text().splitlines())
            if row['split']=='consumed_false_alarm'}
for case in p['cases']:
    part=get_part(case['part']);nominal=part.layers()
    failure=random_failure(nominal,np.random.default_rng(case['seed']),kind='under_extrusion')
    with MujocoPrinter(nominal,failure,case['seed'],case['profile'],False) as printer:
        additions=[];add=printer._add_layer_visual
        def traced_add(layer,color,thickness):
            start=printer.renderer.scene.ngeom
            add(layer,color,thickness)
            additions.append((start,printer.renderer.scene.ngeom))
        printer._add_layer_visual=traced_add
        for index in range(max(case['layers'])+1):
            actual=printer.print_layer();additions.clear();image=printer.camera_image()
            if index not in case['layers']:continue
            ref=reference[case['name'],index]
            assert camera_frame_digest(image).hex()==ref['image_sha256'][:32]
            rng_before=json.dumps(printer.rng_camera.bit_generator.state,sort_keys=True)
            state_before=printer.data.qpos.copy();frames_before=printer.camera_frames
            current_start,current_end=additions[-1]
            assert current_end==printer.renderer.scene.ngeom
            original_rgb=printer.renderer.render()
            full_visible=visibility(printer,current_start)
            assert np.array_equal(original_rgb,printer.renderer.render()),'diagnostic labels changed RGB'
            # Re-render the same last scene with only the current deposited
            # layer. This is an offline visibility ablation, unavailable to a
            # runtime policy and never counted as an improved camera input.
            printer.renderer.update_scene(printer.data,camera='inspection')
            only_start=printer.renderer.scene.ngeom
            add(actual,np.asarray(printer.profile.filament_rgb),.25)
            only_rgb=printer.renderer.render()[:,:,::-1].copy()
            only_image=printer._calibration.top_down(only_rgb,actual.z,(0,0))
            only_visible=visibility(printer,only_start)
            nominal_mask=camera.expected_mask(nominal[index]);actual_mask=camera.expected_mask(actual)
            roi=cv2.dilate((nominal_mask!=actual_mask).astype('uint8'),np.ones((9,9),'uint8')).astype(bool)
            full=full_visible>127;only=only_visible>127;hidden=only&~full
            result=dict(case=case['name'],layer=index,image_sha256_128=camera_frame_digest(image).hex(),
                original_image_reproduced=True,diagnostic_segmentation_not_a_model_input=True,
                full_scene_current_visible_pixels=int(full.sum()),current_only_visible_pixels=int(only.sum()),
                hidden_by_older_scene_pixels=int(hidden.sum()),
                hidden_fraction=float(hidden.sum()/max(1,only.sum())),
                deformed_roi_current_only_pixels=int((only&roi).sum()),
                deformed_roi_hidden_pixels=int((hidden&roi).sum()),
                deformed_roi_hidden_fraction=float((hidden&roi).sum()/max(1,(only&roi).sum())),
                truth_geometry=dict(hole_area_mm2=F.base(actual)['hole_area'],open_wall=ref['truth']['open_wall']),
                truth_raster=enclosed(actual_mask*255,nominal[index]),
                scene_raster=enclosed(camera.expected_mask(Layer(actual.z,actual.region.buffer(.02).buffer(-.02)))*255,nominal[index]),
                full_scene_current_visibility=enclosed(full_visible,nominal[index]),
                current_only_visibility=enclosed(only_visible,nominal[index]))
            # Keep the physical trajectory and noise stream exactly as captured.
            assert np.array_equal(state_before,printer.data.qpos) and frames_before==printer.camera_frames
            assert json.dumps(printer.rng_camera.bit_generator.state,sort_keys=True)==rng_before
            rows.append(result)
            if index==case['snapshot_layer']:
                for label,value in [('original',image),('current-only',only_image),
                    ('visible-full',full_visible),('visible-current-only',only_visible),('truth-raster',actual_mask*255)]:
                    assert cv2.imwrite(str(root/(case['name']+'-'+label+'.png')),value)
            (root/'frames.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(result),flush=True)
verify()
result=dict(status='complete',frames=len(rows),source_sha256=p['source_sha256'],
    original_images_reproduced=True,physical_state_and_camera_rng_unchanged=True,
    frames_sha256=digest(root/'frames.json'),protocol_sha256=digest(root/'protocol.json'),
    new_training_runs=0,new_recovery_trials=0,limitation=p['limitation'])
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
