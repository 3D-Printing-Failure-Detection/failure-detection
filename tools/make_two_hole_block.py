"""Write the STL for the first real-printer test part: a 50 x 10 x 20 mm block with two
vertical M5 clearance holes 30 mm apart (parts/print/two-hole-block/).

  python tools/make_two_hole_block.py

The cross-section is built with shapely exactly as the part YAML describes it, then extruded,
so the printed part and the pipeline's nominal geometry are the same shape.
"""
from pathlib import Path

import trimesh
from shapely.geometry import Point, box

LENGTH, WIDTH, HEIGHT = 50.0, 10.0, 20.0
HOLE_D, PITCH = 5.5, 30.0                 # ISO 273 medium clearance for M5

OUT = Path(__file__).resolve().parent.parent / "parts" / "print" / "two-hole-block" / "two-hole-block.stl"


def section():
    s = box(-LENGTH / 2, -WIDTH / 2, LENGTH / 2, WIDTH / 2)
    for x in (-PITCH / 2, PITCH / 2):
        s = s.difference(Point(x, 0).buffer(HOLE_D / 2, quad_segs=32))
    return s


def main():
    mesh = trimesh.creation.extrude_polygon(section(), HEIGHT)
    # bed origin at the front-left-bottom corner (datum corner), so slicer coordinates are positive
    mesh.apply_translation([LENGTH / 2, WIDTH / 2, 0])
    assert mesh.is_watertight and mesh.is_volume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT)
    print(f"{OUT}: {len(mesh.faces)} faces, volume {mesh.volume:.0f} mm3, bounds {mesh.bounds.tolist()}")


if __name__ == "__main__":
    main()
