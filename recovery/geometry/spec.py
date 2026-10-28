"""Function specs: turn a measured/planned layer stack into a pass/fail verdict.

Checks (all optional, driven by parts/<name>.yaml):
  container    closed walls above base_h, flat solid base, capacity range
  min_wall_mm  no layer (beyond a small defect budget) thinner than this
  material_volume_mm3, holes (through-hole size/position)
  connected (z-range must be N pieces, e.g. handle fused), present (material must exist)
The checks only look at layer cross-sections, so any part is described by data alone.
Always on: excess overhang vs. the nominal design (a layer that hangs in air
where the design is supported will not bond).
"""
from dataclasses import dataclass, field
from typing import Dict, List

from shapely.geometry import Point

from ..types import Layer
from . import features as F
from .ops import polys

MAX_DEFECT_LAYERS = 3        # layers allowed to dip below min_wall
HARD_WALL_FRAC = 0.6         # ...but never below this fraction of min_wall
OVERHANG_AREA = 6.0          # mm^2 of excess overhang that breaks a part


@dataclass
class Verdict:
    ok: bool
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return "OK" if self.ok else "FAIL: " + "; ".join(self.violations)


def check(layers: List[Layer], planned: List[Layer], spec: dict, layer_h: float) -> Verdict:
    assert len(layers) == len(planned), (len(layers), len(planned))
    viol, met = [], {}

    # --- excess overhang ---
    worst = 0.0
    for j in range(1, len(layers)):
        worst = max(worst, F.excess_overhang(layers[j], layers[j - 1], planned[j], planned[j - 1]))
    met["max_overhang_mm2"] = worst
    if worst > OVERHANG_AREA:
        viol.append(f"layer not supported by layer below ({worst:.0f} mm2 overhang)")

    # --- container: closed walls, flat base, capacity ---
    c = spec.get("container")
    if c:
        base_h = c["base_h"]
        cap, bad_wall, bad_base = 0.0, 0, 0
        for lay, pl in zip(layers, planned):
            f, pf = F.base(lay), F.base(pl)
            if lay.z <= base_h:
                if f["area"] < 0.85 * pf["area"] or f["n_main_holes"] > 0:
                    bad_base += 1
            else:
                if f["hole_area"] < 0.5 * pf["hole_area"]:
                    bad_wall += 1
                cap += f["hole_area"] * layer_h
        met["capacity_mm3"] = cap
        if bad_wall:
            viol.append(f"walls not closed in {bad_wall} layers (leaks)")
        if bad_base:
            viol.append(f"base not flat/solid in {bad_base} layers")
        lo, hi = c["capacity_mm3"]
        if not (lo <= cap <= hi):
            viol.append(f"capacity {cap/1000:.1f} ml outside [{lo/1000:.0f},{hi/1000:.0f}]")

    # --- minimum wall thickness ---
    t = spec.get("min_wall_mm")
    if t:
        soft = hard = 0
        for lay, pl in zip(layers, planned):
            if F.thin_frac(lay, t) - F.thin_frac(pl, t) > 0.03:
                soft += 1
            if F.thin_frac(lay, HARD_WALL_FRAC * t) - F.thin_frac(pl, HARD_WALL_FRAC * t) > 0.03:
                hard += 1
        met["thin_layers"] = soft
        if hard:
            viol.append(f"wall far below {t} mm in {hard} layers")
        elif soft > MAX_DEFECT_LAYERS:
            viol.append(f"wall below {t} mm in {soft} layers")

    # --- material volume ---
    mv = spec.get("material_volume_mm3")
    if mv:
        vol = sum(l.region.area for l in layers) * layer_h
        met["material_mm3"] = vol
        if not (mv[0] <= vol <= mv[1]):
            viol.append(f"material volume {vol:.0f} mm3 outside {mv}")

    # --- through holes ---
    for hs in spec.get("holes", []):
        inter, ok = None, True
        for lay in layers:
            cand = F.hole_near(lay, hs["center"], hs["diameter"][1])
            if not cand:
                ok = False
                break
            h = max(cand, key=lambda p: p.area)
            inter = h if inter is None else inter.intersection(h)
            if inter.is_empty:
                ok = False
                break
        if ok:
            ps = polys(inter)
            ok = bool(ps)
        if not ok:
            viol.append(f"through-hole at {hs['center']} blocked")
            continue
        (cx, cy), d = F.inscribed(max(ps, key=lambda p: p.area))
        dc = Point(cx, cy).distance(Point(*hs["center"]))
        met[f"hole_{hs['center']}_d"] = d
        if not (hs["diameter"][0] <= d <= hs["diameter"][1]):
            viol.append(f"hole at {hs['center']} diameter {d:.1f} outside {hs['diameter']}")
        elif dc > hs["tol_center"]:
            viol.append(f"hole at {hs['center']} off by {dc:.1f} mm")

    # --- connectivity: parts that must stay joined (a handle, a boss, a rib) ---
    for rule in spec.get("connected", []):
        lo, hi = rule["z"]
        want = rule.get("components", 1)
        for lay in layers:
            if lo <= lay.z <= hi and F.base(lay)["n"] != want:
                viol.append(f"expected {want} piece(s) near z={lay.z:.1f}, found {F.base(lay)['n']}")
                break

    # --- features that must still exist (material not lost) ---
    for rule in spec.get("present", []):
        lo, hi = rule["z"]
        for lay, pl in zip(layers, planned):
            if lo <= lay.z <= hi and lay.region.area < rule["min_area_frac"] * pl.region.area:
                viol.append(f"feature missing near z={lay.z:.1f}")
                break

    return Verdict(not viol, viol, met)
