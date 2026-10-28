"""Spec author: pin down the FORM and FUNCTION of any part, dynamically.

  form      the geometry that must survive a failure: topology per height band (which pieces
            must be joined, which features must exist), through-holes, wall thickness, volume
  function  what the part does: holds liquid (capacity, no leaks), carries loads (load cases
            with forces, supports and safety factors), fits bolts

An LLM (Claude) reads
  (1) a plain-English description of the part and
  (2) a measured summary of its sliced geometry (bounding box, cavity, holes, bands...)
and writes the spec. It never gets to trust itself: every proposal is VERIFIED against the
nominal part by the geometry checker and the FE stress simulator, and the violations are fed
back for up to `rounds` repairs. With no API key (or --no-llm) a deterministic heuristic
author produces a conservative spec from the same summary, so the pipeline works offline.

  python -m recovery.cli specgen --part parts/mypart.yaml --describe "a phone stand ..."
"""
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import yaml

from .fea import MATERIALS, stress_test
from .fea.cases import liquid_cavity
from .fea.voxel import stack_bounds, voxelize
from .geometry import features as F
from .geometry.ops import hole_polys, polys
from .geometry.spec import check
from .types import Layer

SCHEMA_DOC = """
Return ONE JSON object: {"rationale": "<2-4 sentences: what the part is for and why these limits>",
 "spec": { ...keys below, all optional, include only what this part needs... }}

Coordinates are millimetres, part-centred in x/y, z measured from the bed.

form keys
  min_wall_mm: number                       no wall thinner than this
  material_volume_mm3: [lo, hi]
  holes: [{center:[x,y], diameter:[lo,hi], tol_center: number}]      through-holes (bolts, shafts)
  connected: [{z:[a,b], components: N}]     the cross-section must be exactly N pieces in this z-range
  present:   [{z:[a,b], min_area_frac: 0.85}]   material must still exist here
function keys
  container: {base_h: number, capacity_mm3: [lo, hi]}     closed walls, flat solid base, capacity
  material: "PLA" | "PETG" | "ABS"
  loads: [ {name, fluid?, supports?, loads?, self_weight?, require} ]   stress tests, where
     fluid: {density_kg_m3, fill_height_mm}                  liquid in any enclosed cavity (also tests for leaks)
     supports: [{face: "bottom"|"top"} | {box:[xmin,xmax,ymin,ymax,zmin,zmax]}]   fixed regions (default bottom face)
     loads: [{face|box, force_N, direction:[dx,dy,dz]}]      force spread over the selected surface/nodes
     require: {safety_factor: number (>=1.5), max_deflection_mm?: number, leak_free?: bool}
Printed parts are anisotropic: layers are weakest in tension ACROSS layers (z). Choose realistic
loads for the part's job (a held mug ~30 N on the handle, a clamped spacer ~ bolt preload, ...),
and put load boxes on the real features listed in the geometry summary. Tolerances should leave
useful slack (a mug that holds 10% less is still a mug) but never allow a non-functional part.
"""

ALLOWED = {"min_wall_mm", "material_volume_mm3", "holes", "connected", "present", "container",
           "material", "loads"}


def _py(o):
    """numpy scalars -> plain Python (JSON / YAML cannot serialise numpy types)."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


# ---------------------------------------------------------------- geometry summary
def geometry_summary(layers: List[Layer], layer_h: float) -> dict:
    nz = len(layers)
    minx, miny, maxx, maxy = stack_bounds(layers)
    height = layers[-1].z
    s = dict(bbox_mm=dict(x=[round(minx, 1), round(maxx, 1)], y=[round(miny, 1), round(maxy, 1)],
                          z=[0, round(height, 2)]),
             layer_height_mm=round(layer_h, 3), n_layers=nz,
             material_volume_mm3=round(sum(l.region.area for l in layers) * layer_h))

    # topology per height band (e.g. a handle: 1 piece, then 2, then 1 again)
    runs, start = [], 0
    ncomp = [F.base(l)["n"] for l in layers]
    for i in range(1, nz + 1):
        if i == nz or ncomp[i] != ncomp[start]:
            seg = layers[start:i]
            runs.append(dict(z=[round(seg[0].z - layer_h, 2), round(seg[-1].z, 2)], pieces=ncomp[start],
                             area_mm2=round(float(np.mean([l.region.area for l in seg])), 1)))
            start = i
    s["bands"] = runs

    # cavity that holds liquid (flood-fill test at 90% height)
    fine = voxelize(layers, 1.0, max(1, int(round(1.0 / layer_h))), "min")
    cav = liquid_cavity(fine.solid, fine.hz, fine.z0, 0.9 * height)
    cap = float(cav.sum() * fine.h ** 2 * fine.hz)
    s["cavity"] = None
    if cap > 500:
        ks = np.nonzero(cav.any((0, 1)))[0]
        xs = np.nonzero(cav.any((1, 2)))[0]
        ys = np.nonzero(cav.any((0, 2)))[0]
        s["cavity"] = dict(capacity_mm3=round(cap), floor_thickness_mm=round(float(ks.min() * fine.hz), 2),
                           x=[round(fine.x0 + xs.min() * fine.h, 1), round(fine.x0 + (xs.max() + 1) * fine.h, 1)],
                           y=[round(fine.y0 + ys.min() * fine.h, 1), round(fine.y0 + (ys.max() + 1) * fine.h, 1)])

    # through-holes: an interior hole present (same place) in every layer
    through, mid = [], layers[nz // 2]
    sample = layers[:: max(1, nz // 24)]
    for hp in hole_polys(mid.region):
        pt = hp.representative_point()
        inter, ok = hp, True
        for lay in sample:
            cands = [q for q in hole_polys(lay.region) if q.contains(pt)]
            if not cands:
                ok = False
                break
            inter = inter.intersection(cands[0])
        ps = polys(inter) if ok else []
        if ps:
            (cx, cy), d = F.inscribed(max(ps, key=lambda p: p.area))
            through.append(dict(center=[round(cx, 1), round(cy, 1)], diameter_mm=round(d, 2)))
    s["through_holes"] = through

    # thinnest wall: largest opening size that removes < 2% of material, over sampled layers
    est = []
    for lay in layers[:: max(1, nz // 12)]:
        if lay.region.is_empty:
            continue
        t_ok = 0.4
        for t in np.arange(0.4, 12.01, 0.4):
            if F.thin_frac(lay, float(t)) <= 0.02:
                t_ok = float(t)
            else:
                break
        est.append(t_ok)
    s["thin_wall_estimate_mm"] = round(float(np.percentile(est, 10)), 1) if est else None
    return _py(s)


# ---------------------------------------------------------------- validation
def validate_spec(spec: dict, summary: dict) -> List[str]:
    """Structural errors in a proposed spec (empty list = well-formed)."""
    errs = []
    if not isinstance(spec, dict):
        return ["spec must be an object"]
    for k in spec:
        if k not in ALLOWED:
            errs.append(f"unknown key '{k}'")
    bb = summary["bbox_mm"]
    H = bb["z"][1]

    def rng(v, name):
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] < v[1]):
            errs.append(f"{name} must be [lo, hi] with lo < hi")

    if "material" in spec and spec["material"] not in MATERIALS:
        errs.append(f"material must be one of {list(MATERIALS)}")
    if "min_wall_mm" in spec and not (0.3 <= spec["min_wall_mm"] <= 20):
        errs.append("min_wall_mm out of range")
    if "material_volume_mm3" in spec:
        rng(spec["material_volume_mm3"], "material_volume_mm3")
    if "container" in spec:
        c = spec["container"]
        rng(c.get("capacity_mm3"), "container.capacity_mm3")
        if not (0 <= c.get("base_h", -1) < H):
            errs.append("container.base_h must be within the part height")
    for i, hs in enumerate(spec.get("holes", [])):
        rng(hs.get("diameter"), f"holes[{i}].diameter")
        if "center" not in hs or "tol_center" not in hs:
            errs.append(f"holes[{i}] needs center and tol_center")
    for key in ("connected", "present"):
        for i, r in enumerate(spec.get(key, [])):
            if "z" not in r or not (0 <= r["z"][0] < r["z"][1] <= H + 0.01):
                errs.append(f"{key}[{i}].z must be inside [0, {H}]")
    for i, case in enumerate(spec.get("loads", [])):
        if "require" not in case:
            errs.append(f"loads[{i}] needs 'require'")
        elif case["require"].get("safety_factor", 1.5) < 1.2:
            errs.append(f"loads[{i}] safety_factor too low (use >= 1.5)")
        for sel in case.get("supports", []) + case.get("loads", []):
            if not ("face" in sel or "box" in sel):
                errs.append(f"loads[{i}]: every selector needs 'face' or 'box'")
            if "box" in sel and len(sel["box"]) != 6:
                errs.append(f"loads[{i}]: box needs 6 numbers")
        for ld in case.get("loads", []):
            if "force_N" not in ld:
                errs.append(f"loads[{i}]: a load needs force_N")
    return errs


def verify_spec(spec: dict, layers: List[Layer], layer_h: float, h=1.5) -> Tuple[bool, List[str], Optional[object]]:
    """The nominal part must satisfy its own spec (geometry and stress tests)."""
    v = check(layers, layers, spec, layer_h)
    msgs = list(v.violations)
    report = None
    if spec.get("loads"):
        report = stress_test(layers, spec, layer_h, h)
        msgs += report.violations
    return (not msgs), msgs, report


# ---------------------------------------------------------------- authors
def heuristic_spec(summary: dict, layers: List[Layer], layer_h: float) -> dict:
    """Deterministic fallback: derive conservative form + function from the summary alone."""
    spec = {"material": "PLA"}
    vol = summary["material_volume_mm3"]
    spec["material_volume_mm3"] = [round(0.85 * vol), round(1.15 * vol)]
    tw = summary.get("thin_wall_estimate_mm")
    if tw:
        spec["min_wall_mm"] = max(0.8, round(0.7 * tw, 1))
    H = summary["bbox_mm"]["z"][1]
    cav = summary.get("cavity")
    if cav:
        base_h = cav["floor_thickness_mm"]
        spec["container"] = dict(base_h=base_h, capacity_mm3=[round(0.75 * cav["capacity_mm3"]),
                                                              round(1.25 * cav["capacity_mm3"])])
    for th in summary.get("through_holes", []):
        d = th["diameter_mm"]
        spec.setdefault("holes", []).append(dict(center=th["center"], diameter=[round(0.85 * d, 1),
                                                                                  round(1.2 * d, 1)],
                                                 tol_center=1.0))
    # form: keep the nominal topology in every band at least 2 mm tall
    bands = [b for b in summary["bands"] if b["z"][1] - b["z"][0] >= 2.0]
    if len(summary["bands"]) > 1:
        spec["connected"] = [dict(z=[round(b["z"][0] + 0.5, 2), round(b["z"][1] - 0.5, 2)], components=b["pieces"])
                             for b in bands]
    loads = []
    if cav:
        loads.append(dict(name="full of water on a table",
                          fluid=dict(density_kg_m3=1000, fill_height_mm=round(0.9 * H, 1)),
                          supports=[dict(face="bottom")], require=dict(safety_factor=1.5, leak_free=True)))
    else:
        loads.append(dict(name="compression, 500 N on the top face", supports=[dict(face="bottom")],
                          loads=[dict(face="top", force_N=500, direction=[0, 0, -1])],
                          require=dict(safety_factor=1.5)))
    spec["loads"] = loads
    return _py(spec)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def author_spec(layers: List[Layer], layer_h: float, description: str = "", use_llm: bool = True,
                rounds: int = 3, client=None, model: Optional[str] = None, h=1.5, log=print) -> dict:
    """-> {"spec": {...}, "rationale": str, "source": "llm"|"heuristic", "verified": bool,
           "issues": [...], "summary": {...}}"""
    summary = geometry_summary(layers, layer_h)
    fallback = lambda why: _heuristic_result(summary, layers, layer_h, h, why)
    if not use_llm:
        return fallback("LLM disabled")
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        from .loop.llm import MODEL
        model = model or MODEL
    except Exception as e:                                  # no SDK / no key
        return fallback(f"LLM unavailable ({type(e).__name__})")

    messages = [{"role": "user", "content":
                 f"PART DESCRIPTION:\n{description or '(none given: infer the likely job from the geometry)'}\n\n"
                 f"GEOMETRY SUMMARY (measured from the sliced part):\n{json.dumps(summary, indent=1)}\n\n"
                 f"{SCHEMA_DOC}"}]
    issues: List[str] = []
    spec, rationale = None, ""
    for r in range(rounds):
        try:
            msg = client.messages.create(model=model, max_tokens=3000,
                                         system="You are a mechanical design engineer writing the acceptance "
                                                "spec for a 3D-printed part. Reply with JSON only.",
                                         messages=messages)
            text = msg.content[0].text
            out = _extract_json(text)
            spec, rationale = out["spec"], out.get("rationale", "")
        except Exception as e:
            return fallback(f"LLM reply unusable ({type(e).__name__}: {e})")
        errs = validate_spec(spec, summary)
        if errs:
            issues = errs
        else:
            ok, msgs, _ = verify_spec(spec, layers, layer_h, h)
            issues = msgs
            if ok:
                log(f"spec verified against the nominal part after {r+1} round(s)")
                return dict(spec=spec, rationale=rationale, source="llm", verified=True, issues=[],
                            summary=summary)
        log(f"round {r+1}: {len(issues)} problem(s); asking the LLM to repair")
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Your spec failed verification against the NOMINAL part "
                      "(a correct spec must pass on the part as designed):\n- " + "\n- ".join(issues[:12])
                      + "\nFix the spec (not the part). Reply with the full JSON again."}]
    res = fallback(f"LLM spec did not verify after {rounds} rounds")
    res["issues"] = issues
    return res


def _heuristic_result(summary, layers, layer_h, h, why):
    spec = heuristic_spec(summary, layers, layer_h)
    ok, msgs, _ = verify_spec(spec, layers, layer_h, h)
    return dict(spec=spec, rationale=f"Heuristic spec ({why}): nominal topology, +-15-25% tolerances, "
                                     "one default load case.", source="heuristic", verified=ok,
                issues=msgs, summary=summary)


def save_spec(result: dict, path, name: str):
    doc = dict(name=name, source=result["source"], verified=result["verified"],
               rationale=result["rationale"], **result["spec"])
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=100)
