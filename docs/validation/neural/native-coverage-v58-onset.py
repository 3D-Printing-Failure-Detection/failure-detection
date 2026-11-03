from pathlib import Path
import hashlib,json,sys
root=Path('/mnt/data/ishaan/hri-workspace/failure-detection/neural-native-coverage-v58')
read=lambda path:json.loads(Path(path).read_text())
digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(root/'protocol.json');sys.path.insert(0,p['source'])
import cv2
from recovery import camera
from recovery.geometry.ops import clean,polys
from recovery.models.boundary import BoundarySegmenter
from recovery.parts import get_part
from recovery.types import Layer
from tools.inspect_camera_case import layer_features
from tools.experiment_provenance import runtime_digest
assert runtime_digest(p['source'])==p['source_sha256']
rows=[]
for case in p['identity_cases']:
    image=cv2.imread(case['image']);previous=cv2.imread(case['previous'])
    part=get_part(case['part']);nominal=part.layers();index=case['layer']
    reference=read(Path(p['base_diagnostics'])/case['name']/'layers.json')[index]
    for name,path in [('base',Path(p['base_checkpoint'])),('linear',root/'linear/boundary.npz'),('nonlinear',root/'nonlinear/boundary.npz')]:
        model=BoundarySegmenter(path)
        model.prime_history(previous,camera.expected_mask(nominal[index-1]))
        region=clean(model.measure(image,camera.expected_mask(nominal[index])).simplify(.25/4))
        features=layer_features(Layer(nominal[index].z,region),nominal[index],part.spec)
        if name=='base':
            expected={k:v for k,v in reference['measured']['combined'].items() if k not in ('iou','boundary_error_mm')}
            assert features==expected,'base onset features did not reproduce'
        holes=sorted((__import__('shapely').Polygon(ring).area for poly in polys(region) for ring in poly.interiors),reverse=True)
        rows.append(dict(case=case['name'],layer=index,model=name,checkpoint_sha256=model.sha256,
            current_image_sha256=digest(case['image']),previous_image_sha256=digest(case['previous']),
            features=features,hole_areas_mm2=holes,truth=reference['truth'],
            region_sha256=hashlib.sha256(region.wkb).hexdigest()))
result=dict(status='complete',source_sha256=p['source_sha256'],script_sha256=digest(__file__),
    protocol_sha256=digest(root/'protocol.json'),rows=rows,all_base_onset_features_reproduced=True,
    limitation='Fixed checkpoint screen on four consumed onset images; no checkpoint selection, functional-stack qualification or recovery result.')
out=root/'onset-screen.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
for row in rows:
    f=row['features'];print(row['case'],row['model'],'holes',len(row['hole_areas_mm2']),
        'gap',f.get('open_wall'),'hard_thin',f.get('hard_thin'),'excess',f.get('hard_thin_excess'))
