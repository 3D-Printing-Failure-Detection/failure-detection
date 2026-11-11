"""Write the STL for test part 2: a 30 x 20 x 12 mm open tray with 1.2 mm walls and floor
(parts/print/water-tray/).

  python tools/make_water_tray.py          # needs trimesh + manifold3d

Same shape as the part YAML's CSG: a full block with the cavity cut from z = 1.2 mm up.
"""
from pathlib import Path

import trimesh

LENGTH, WIDTH, HEIGHT = 30.0, 20.0, 12.0
WALL = FLOOR = 1.2                        # 3 lines of a 0.42 mm wall

OUT = Path(__file__).resolve().parent.parent / "parts" / "print" / "water-tray" / "water-tray.stl"


def main():
    outer = trimesh.creation.box((LENGTH, WIDTH, HEIGHT))
    outer.apply_translation([0, 0, HEIGHT / 2])
    cavity = trimesh.creation.box((LENGTH - 2 * WALL, WIDTH - 2 * WALL, HEIGHT))   # pokes out the top
    cavity.apply_translation([0, 0, FLOOR + HEIGHT / 2])
    mesh = outer.difference(cavity)
    mesh.apply_translation([LENGTH / 2, WIDTH / 2, 0])     # datum corner at the origin
    assert mesh.is_watertight and mesh.is_volume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(OUT)
    print(f"{OUT}: {len(mesh.faces)} faces, volume {mesh.volume:.0f} mm3, bounds {mesh.bounds.tolist()}")


if __name__ == "__main__":
    main()
