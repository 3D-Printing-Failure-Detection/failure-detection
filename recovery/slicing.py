"""Get the nominal layers of a part: from Bambu Studio's sliced output (what the P1S will
actually print; preferred), or by cross-sectioning an STL ourselves with trimesh."""
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from .gcode import gcode_to_layers
from .geometry.ops import clean, translate
from .types import Layer


def stl_to_layers(stl_path, layer_h=0.2, centre=True) -> List[Layer]:
    """Cross-section an STL at mid-layer heights. Works for any closed mesh; use for
    2.5D parts (cups, brackets...) whose function is set by cross-sections."""
    import trimesh
    mesh = trimesh.load(stl_path, force="mesh")
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds
    cx, cy = ((x0 + x1) / 2, (y0 + y1) / 2) if centre else (0.0, 0.0)
    layers = []
    n = int(round((z1 - z0) / layer_h))
    for i in range(n):
        z = z0 + (i + 0.5) * layer_h
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            layers.append(Layer(round((i + 1) * layer_h, 4), MultiPolygon()))
            continue
        planar, _ = sec.to_planar(to_2D=np.eye(4))
        region = unary_union(planar.polygons_full) if len(planar.polygons_full) else MultiPolygon()
        layers.append(Layer(round((i + 1) * layer_h, 4), clean(translate(region, -cx, -cy))))
    return layers


BAMBU_STUDIO_MAC = "/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"


def read_gcode_3mf(path, plate=1) -> str:
    """G-code text from a Bambu Studio sliced project (*.gcode.3mf is a zip that holds
    Metadata/plate_<n>.gcode), or from a plain .gcode file."""
    path = Path(path)
    if path.suffix.lower() != ".3mf":
        return path.read_text()
    with zipfile.ZipFile(path) as z:
        name = f"Metadata/plate_{plate}.gcode"
        if name not in z.namelist():
            found = [n for n in z.namelist() if n.endswith(".gcode")]
            if not found:
                raise FileNotFoundError(f"no sliced G-code in {path} (export 'Plate sliced file')")
            name = found[0]
        return z.read(name).decode("utf-8", errors="replace")


def bambu_studio_slice(model, out_3mf, settings=None, filaments=None, plate=1,
                       exe=BAMBU_STUDIO_MAC, extra_args=()):
    """Slice with Bambu Studio's command line. `settings` = (machine.json, process.json)
    exported from the GUI for a P1S; `filaments` = filament.json list. Returns the G-code text.

    Equivalent to:
      BambuStudio --slice 1 --load-settings "machine.json;process.json" \
                  --load-filaments "filament.json" --allow-newer-file --export-3mf out.gcode.3mf model
    Recommended process settings for this project: skirt loops 0, brim off, no supports,
    'arc fitting' off, a plain wall/infill pattern, and timelapse 'traditional' so the head
    parks at every layer change (clear camera view).
    """
    if shutil.which(exe) is None and not Path(exe).exists():
        raise FileNotFoundError(f"Bambu Studio not found at {exe}; pass exe=...")
    cmd = [exe, "--slice", str(plate), "--allow-newer-file", "--export-3mf", str(out_3mf)]
    if settings:
        cmd += ["--load-settings", ";".join(map(str, settings))]
    if filaments:
        cmd += ["--load-filaments", ";".join(map(str, filaments))]
    cmd += list(extra_args) + [str(model)]
    subprocess.run(cmd, check=True, capture_output=True)
    return read_gcode_3mf(out_3mf, plate)
