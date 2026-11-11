"""Write the STL for test part 3: a 4-lobe knob, 31 mm across and 8 mm tall, with a hex hole
straight through for a pressed-in M5 nut (parts/print/star-knob/).

  python tools/make_star_knob.py

The hex goes all the way through so the existing `holes` spec rule can measure it: the largest
circle that fits a hexagon is its across-flats size. The part YAML loads this STL (`stl:`), so
the pipeline's nominal layers are exactly this shape.
"""
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

HEIGHT = 8.0
HUB_R = 6.5                    # hub radius: leaves ~2.3 mm of wall between a valley and the hex
LOBES, LOBE_R, LOBE_AT = 4, 5.5, 10.0     # 4 lobes of radius 5.5 centred 10 mm out -> 31 mm across;
                                          # wide valleys reach down to the hub (6 lobes merge into notches)
FILLET = 1.0                   # rounds the valleys between lobes
HEX_AF = 8.2                   # M5 nut is 8.0 across flats; prints come out ~0.1-0.2 mm small

OUT = Path(__file__).resolve().parent.parent / "parts" / "print" / "star-knob" / "star-knob.stl"


def hexagon(af):
    r = af / np.sqrt(3)                                    # circumradius
    ang = np.deg2rad(np.arange(0, 360, 60))
    return Polygon(np.c_[r * np.cos(ang), r * np.sin(ang)])


def section():
    lobes = [Point(LOBE_AT * np.cos(a), LOBE_AT * np.sin(a)).buffer(LOBE_R, quad_segs=24)
             for a in np.deg2rad(np.arange(0, 360, 360 / LOBES))]
    outline = unary_union([Point(0, 0).buffer(HUB_R, quad_segs=24), *lobes])
    outline = outline.buffer(FILLET, quad_segs=8).buffer(-FILLET, quad_segs=8)   # fillet the valleys
    return outline.difference(hexagon(HEX_AF))


def main():
    mesh = trimesh.creation.extrude_polygon(section(), HEIGHT)
    assert mesh.is_watertight and mesh.is_volume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT)
    print(f"{OUT}: {len(mesh.faces)} faces, volume {mesh.volume:.0f} mm3, bounds {mesh.bounds.tolist()}")


if __name__ == "__main__":
    main()
