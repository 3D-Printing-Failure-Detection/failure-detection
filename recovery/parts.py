"""Parts are DATA, not code. A part is a YAML file holding its geometry and its function spec;
nothing in the pipeline (detector, measurement, assess, decide, adapt) knows about any
particular part.

Two ways to describe geometry in the YAML:

  1. A CSG stack of extruded shapes (2.5D), applied in order, each over a z-range:
        height: 40
        layer_h: 0.25
        geometry:
          - add: {circle: {r: 25}}                         # whole height
          - cut: {circle: {r: [20.3, 26]}, z: [2, 40]}     # r:[a,b] tapers linearly over z
          - add: {rect: {center: [32, 0], size: [17, 8]}, z: [8, 14]}
          - cut: {polygon: {points: [[0,0],[5,0],[0,5]]}}
     shapes: circle {center, r}, rect {center, size}, polygon {points}

  2. Bambu Studio's sliced output (what the P1S will actually print; preferred for real runs):
        gcode: path/to/part.gcode.3mf      # or a plain .gcode
     The layers are parsed from the G-code and recentred on the part; `Part.origin` remembers
     where the part sits on the bed so adapted layers are written back to the right place.
  3. A real mesh:    stl: path/to/part.stl      (cross-sectioned with trimesh)

Spec keys (all optional) are documented in geometry/spec.py. Add a new part by dropping a
YAML file into parts/ (or pass any path with --part).
"""
from pathlib import Path
from typing import List

import yaml
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box

from . import config as C
from .types import Layer

SPEC_DIR = Path(__file__).resolve().parent.parent / "parts"


def _lerp(v, t):
    """v is a number, or [a, b] interpolated by t in [0, 1]."""
    if isinstance(v, (list, tuple)):
        return v[0] + (v[1] - v[0]) * t
    return v


def _shape(spec: dict, t: float):
    (kind, p), = spec.items()
    if kind == "circle":
        return Point(*p.get("center", [0, 0])).buffer(_lerp(p["r"], t), quad_segs=32)
    if kind == "rect":
        cx, cy = p.get("center", [0, 0])
        w, h = (_lerp(s, t) for s in p["size"])
        return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    if kind == "polygon":
        return Polygon(p["points"])
    raise ValueError(f"unknown shape '{kind}'")


def _csg_fn(geometry: List[dict], height: float):
    items = []
    for it in geometry:
        op = "add" if "add" in it else "cut"
        z0, z1 = it.get("z", [0.0, height])
        items.append((op, it[op], z0, z1))

    def fn(z):
        g = Polygon()
        for op, shape, z0, z1 in items:
            if not (z0 <= z < z1):
                continue
            s = _shape(shape, (z - z0) / max(z1 - z0, 1e-9))
            g = g.union(s) if op == "add" else g.difference(s)
        return g
    return fn


class Part:
    """name, height, layer_h, spec, and layers() -> nominal layer stack."""

    def __init__(self, name, spec, layer_h=0.25, fn=None, height=None, layers=None,
                 origin=None, gcode_text=None):
        self.name, self.spec, self.layer_h = name, spec, layer_h
        self.fn, self._height, self._layers = fn, height, layers
        self.origin = origin or C.BED_CENTER      # bed position (mm) of the part centre
        self.gcode_text = gcode_text              # original sliced G-code, if any

    @property
    def height(self):
        return self._height if self._layers is None else self._layers[-1].z

    @property
    def n_layers(self):
        return len(self._layers) if self._layers is not None else int(round(self.height / self.layer_h))

    def layers(self) -> List[Layer]:
        if self._layers is None:
            h = self.layer_h
            self._layers = [Layer(round((i + 1) * h, 4), self.fn((i + 0.5) * h))
                            for i in range(self.n_layers)]
        # the camera model has a fixed field of view; fail early if the part does not fit
        half = C.CAM.fov_mm / 2 - 2
        for lay in self._layers[:: max(1, len(self._layers) // 8)]:
            if not lay.region.is_empty and max(abs(v) for v in lay.region.bounds) > half:
                raise ValueError(f"part '{self.name}' exceeds the camera field of view "
                                 f"(+-{half:.0f} mm); increase config.CameraCfg.fov_mm")
        return self._layers

    @classmethod
    def from_yaml(cls, path) -> "Part":
        path = Path(path)
        d = yaml.safe_load(path.read_text())
        name = d.get("name", path.stem)
        lh = float(d.get("layer_h", 0.25))
        if "gcode" in d:
            from .gcode import gcode_to_layers
            from .slicing import read_gcode_3mf
            src = Path(d["gcode"])
            src = src if src.is_absolute() else path.parent / src
            text = read_gcode_3mf(src, d.get("plate", 1))
            raw = gcode_to_layers(text)
            raw = [l for l in raw if not l.region.is_empty]
            minx = min(l.region.bounds[0] for l in raw[:5])
            miny = min(l.region.bounds[1] for l in raw[:5])
            maxx = max(l.region.bounds[2] for l in raw[:5])
            maxy = max(l.region.bounds[3] for l in raw[:5])
            ox, oy = (minx + maxx) / 2, (miny + maxy) / 2         # footprint centre, bed-centred coords
            layers = [Layer(l.z, translate(l.region, -ox, -oy)) for l in raw]
            lh = round(layers[-1].z / len(layers), 4)
            origin = (C.BED_CENTER[0] + ox, C.BED_CENTER[1] + oy)
            return cls(name, d, lh, layers=layers, origin=origin, gcode_text=text)
        if "stl" in d:
            from .slicing import stl_to_layers
            stl = Path(d["stl"])
            stl = stl if stl.is_absolute() else path.parent / stl
            return cls(name, d, lh, layers=stl_to_layers(stl, lh))
        return cls(name, d, lh, fn=_csg_fn(d["geometry"], float(d["height"])), height=float(d["height"]))


def part_names() -> List[str]:
    return sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))


def get_part(name_or_path: str, layer_h: float = None) -> Part:
    """'mug' (parts/mug.yaml) or a path to any part YAML."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = SPEC_DIR / f"{name_or_path}.yaml"
    part = Part.from_yaml(p)
    if layer_h is not None and part._layers is None:
        part.layer_h = layer_h
    return part


PART_NAMES = part_names()
