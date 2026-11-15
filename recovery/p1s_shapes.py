"""Random training shapes for the camera failure detector.

The handful of real parts in parts/ is too few: the detector learned their outlines, and on round parts
(mug, cup, spacer) it read the curved wall as a slight warp or collapse. These are many more shapes, most
of them round (tubes, cups, vases, cylinders, ellipses, rings, domes), plus boxes, polygons and brackets,
in the same CSG format as parts/*.yaml. Each is fixed by its number, so `shape:17` is always the same part.

    from recovery.p1s_shapes import shape_part
    part = shape_part(17)          # a recovery.parts.Part
"""
import numpy as np

from .parts import Part, _csg_fn

KINDS = ("tube", "tapered_tube", "cylinder", "ring", "cup_handle", "ellipse_tube", "dome", "two_tubes",
         "rounded_box", "polygon_tube", "plate_holes", "bracket")
WEIGHTS = np.array([3, 3, 1.5, 1.5, 1.5, 2, 1, 1, 1.5, 1.5, 1, 1], float)   # mostly round


def _ellipse(a, b, n=72, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[round(cx + a * np.cos(s), 3), round(cy + b * np.sin(s), 3)] for s in t]


def _rounded_rect(w, h, r, n=10):
    r = min(r, w / 2 - 0.1, h / 2 - 0.1)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for s in np.linspace(np.radians(a0), np.radians(a0 + 90), n):
            pts.append([round(cx + r * np.cos(s), 3), round(cy + r * np.sin(s), 3)])
    return pts


def _ngon(n, r, rot=0.0):
    return [[round(r * np.cos(rot + 2 * np.pi * k / n), 3), round(r * np.sin(rot + 2 * np.pi * k / n), 3)]
            for k in range(n)]


def shape_spec(i: int) -> dict:
    """The YAML-style spec of training shape i: name, height, layer_h, geometry."""
    rng = np.random.default_rng([20261113, i])
    kind = KINDS[rng.choice(len(KINDS), p=WEIGHTS / WEIGHTS.sum())]
    h = float(np.round(rng.uniform(8, 45), 1))
    lh = float(rng.choice([0.2, 0.2, 0.2, 0.16, 0.24, 0.28]))
    wall = float(np.round(rng.uniform(0.8, 3.0), 2))
    base = float(np.round(rng.uniform(0.6, 3.0), 2))
    R = float(np.round(rng.uniform(8, 32), 1))
    g = []
    if kind == "tube":
        g = [{"add": {"circle": {"r": R}}}, {"cut": {"circle": {"r": R - wall}}, "z": [base, h]}]
        if rng.random() < 0.3:                                   # open at the bottom too
            g[1].pop("z")
    elif kind == "tapered_tube":
        r1 = float(np.round(R * rng.uniform(0.6, 1.4), 1))
        r1 = min(r1, 34.0)
        rb = R + (r1 - R) * base / h                              # outer radius where the cavity starts
        g = [{"add": {"circle": {"r": [R, r1]}}}, {"cut": {"circle": {"r": [rb - wall, r1 - wall]}}, "z": [base, h]}]
    elif kind == "cylinder":
        g = [{"add": {"circle": {"r": R}}}]
        if rng.random() < 0.5:
            g.append({"cut": {"circle": {"r": float(np.round(rng.uniform(1.5, R * 0.4), 1))}}})
    elif kind == "ring":
        h = float(np.round(rng.uniform(3, 12), 1))
        g = [{"add": {"circle": {"r": R}}}, {"cut": {"circle": {"r": R - max(wall, rng.uniform(2, 8))}}}]
    elif kind == "cup_handle":
        R = min(R, 24.0)
        hw = float(np.round(rng.uniform(4, 8), 1))
        g = [{"add": {"circle": {"r": R}}}, {"cut": {"circle": {"r": R - wall}}, "z": [base, h]},
             {"add": {"rect": {"center": [R + 5, 0], "size": [12, hw]}}, "z": [0.25 * h, 0.4 * h]},
             {"add": {"rect": {"center": [R + 5, 0], "size": [12, hw]}}, "z": [0.7 * h, 0.85 * h]},
             {"add": {"rect": {"center": [R + 9, 0], "size": [4, hw]}}, "z": [0.4 * h, 0.7 * h]}]
    elif kind == "ellipse_tube":
        a, b = R, float(np.round(R * rng.uniform(0.45, 0.85), 1))
        g = [{"add": {"polygon": {"points": _ellipse(a, b)}}},
             {"cut": {"polygon": {"points": _ellipse(a - wall, b - wall)}}, "z": [base, h]}]
    elif kind == "dome":
        R = min(R, 25.0)
        h = float(np.round(min(h, R * rng.uniform(0.5, 1.0)), 1))
        top = max(2.0, 0.25 * R)
        g = [{"add": {"circle": {"r": [R, top]}}}]
        rb = R + (top - R) * base / h
        if rng.random() < 0.5 and top - wall > 0.5:
            g.append({"cut": {"circle": {"r": [rb - wall, top - wall]}}, "z": [base, h]})
    elif kind == "two_tubes":
        r = float(np.round(rng.uniform(6, 14), 1))
        d = r + float(rng.uniform(3, 10))
        g = [{"add": {"circle": {"center": [-d, 0], "r": r}}}, {"add": {"circle": {"center": [d, 0], "r": r}}},
             {"cut": {"circle": {"center": [-d, 0], "r": r - wall}}, "z": [base, h]},
             {"cut": {"circle": {"center": [d, 0], "r": r - wall}}, "z": [base, h]}]
        if rng.random() < 0.5:                                   # joined by a bar
            g.insert(2, {"add": {"rect": {"size": [2 * d, 3]}}, "z": [0, min(h, 4)]})
    elif kind == "rounded_box":
        w, d = float(np.round(rng.uniform(15, 55), 1)), float(np.round(rng.uniform(12, 45), 1))
        rr = float(rng.uniform(1, 12))
        g = [{"add": {"polygon": {"points": _rounded_rect(w, d, rr)}}}]
        if rng.random() < 0.7:
            g.append({"cut": {"polygon": {"points": _rounded_rect(w - 2 * wall, d - 2 * wall, max(0.5, rr - wall))}},
                      "z": [base, h]})
    elif kind == "polygon_tube":
        n, rot = int(rng.integers(5, 13)), float(rng.uniform(0, np.pi))
        g = [{"add": {"polygon": {"points": _ngon(n, R, rot)}}},
             {"cut": {"polygon": {"points": _ngon(n, R - wall / np.cos(np.pi / n), rot)}}, "z": [base, h]}]
    elif kind == "plate_holes":
        h = float(np.round(rng.uniform(2, 8), 1))
        w, d = float(np.round(rng.uniform(25, 60), 1)), float(np.round(rng.uniform(15, 40), 1))
        g = [{"add": {"rect": {"size": [w, d]}}}]
        for _ in range(int(rng.integers(1, 5))):
            r = float(np.round(rng.uniform(1.5, 5), 1))
            g.append({"cut": {"circle": {"center": [round(float(rng.uniform(-w / 2 + r + 2, w / 2 - r - 2)), 1),
                                                    round(float(rng.uniform(-d / 2 + r + 2, d / 2 - r - 2)), 1)],
                                         "r": r}}})
    elif kind == "bracket":
        a, b, t = float(rng.uniform(20, 45)), float(rng.uniform(20, 45)), float(rng.uniform(3, 10))
        pts = [[-a / 2, -b / 2], [a / 2, -b / 2], [a / 2, -b / 2 + t], [-a / 2 + t, -b / 2 + t], [-a / 2 + t, b / 2],
               [-a / 2, b / 2]]
        g = [{"add": {"polygon": {"points": [[round(x, 2), round(y, 2)] for x, y in pts]}}}]
    h = round(round(h / lh) * lh, 3)                             # a whole number of layers
    return dict(name=f"shape-{i:03d}-{kind.replace('_', '-')}", height=h, layer_h=lh, geometry=g)


def shape_part(i: int) -> Part:
    s = shape_spec(i)
    return Part(s["name"], s, s["layer_h"], fn=_csg_fn(s["geometry"], s["height"]), height=s["height"])
