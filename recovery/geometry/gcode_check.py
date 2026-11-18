"""Safety check for G-code written by a language model, before it goes to a real printer.

  gcode_check(text, layer, jobs, layer_h, origin)    problems in ONE layer's G-code (empty = safe)
  gcode_file_check(text, layers, origin)             problems in the whole file that is uploaded

Each problem is one short sentence that names the line, so it can be sent back to the model.

Layer check (`jobs` = the planned toolpaths from gcode_writer.layer_jobs: (feature, Nx2 polyline in
part-centred mm, line width)):
  commands     only G0/G1 with X Y Z E F; Z only at the layer height or lifted 0.4 mm; X/Y inside the
               part's area + 2 mm; no extrusion while lifted or on G0; no retract while moving
  paths        every planned path is traced (within 0.1 mm), nothing is extruded off the planned
               paths, in the given order, each once, and an extrusion never jumps from one path to
               another without a travel
  flow         E per mm of every move matches the width of the path it lies on (+-15%); total E
               within 10% of the plan; retracts/primes at most 1 mm and balanced
  travel       a travel of more than 1 mm is lifted and follows a retract; the filament is primed
               again before the next extrusion
  speeds       extrusion, travel, Z and retract/prime moves each inside a band around 1800/7200/600/1800

File check: layers rise in order and match the plan, `M73 L` runs 1, 2, 3 ..., no homing or bed
levelling (a header `G28 X Y` re-home, the project's own opt-in preset, is the one exception),
heaters only in the header (and off in the footer), and the footer parks the head clear of the part.
"""
import re
from typing import List, Optional

import numpy as np
import shapely
from shapely.geometry import LineString

ALLOWED_LETTERS = set("XYZEF")
ZHOP = 0.4
RETRACT_MIN = 0.4            # a travel needs at least this much filament pulled back
TRACE_TOL_MM = 0.1           # a planned path is "traced" if the extrusion is this close
RATE_TOL = 0.15              # per extrusion move (moves of at least MIN_SEGMENT_MM)
PATH_RATE_TOL = 0.10         # over a whole path
MIN_SEGMENT_MM = 1.0
SPEED_BANDS = dict(extrusion=(900, 3600), travel=(3600, 12000), z=(300, 1500), retract=(900, 3600))
MAX_REPORTED = 3


def _e_per_mm(width, layer_h):
    from ..gcode import _e
    return _e(1.0, width, layer_h)


def _samples(line: LineString, step=0.5):
    n = max(int(np.ceil(line.length / step)), 1)
    return shapely.line_interpolate_point(line, np.linspace(0, line.length, n + 1))


def _parse(raw):
    code = raw.split(";", 1)[0].strip()
    if not code:
        return None
    op, *args = code.split()
    return code, op.upper(), args


def gcode_check(text: str, layer, jobs, layer_h: float, origin, margin_mm: float = 2.0) -> List[str]:
    z, problems = layer.z, []
    ox, oy = origin
    x0, y0, x1, y1 = layer.region.bounds
    x0, x1, y0, y1 = x0 + ox - margin_mm, x1 + ox + margin_mm, y0 + oy - margin_mm, y1 + oy + margin_mm
    expected = [_e_per_mm(wd, layer_h) for _, _, wd in jobs]
    want_e = sum(_e_per_mm(wd, layer_h) * LineString(p).length for _, p, wd in jobs)
    job_lines = [LineString(p) for _, p, _ in jobs]
    planned = shapely.union_all(job_lines) if job_lines else None

    x = y = None
    cz = z - layer_h                       # the previous layer's height: the nozzle is on the part
    f_modal = None
    retracted = 0.0                        # filament currently pulled back
    e_still = e_total = 0.0
    segs = []                              # (line no, start, end, e, run id)
    run, in_run = 0, False
    for n, raw in enumerate(text.splitlines(), 1):
        parsed = _parse(raw)
        if parsed is None:
            continue
        code, op, args = parsed
        if op not in ("G0", "G1"):
            problems.append(f"line {n}: '{code}' is not allowed (only G0/G1)")
            in_run = False
            continue
        v, ok = {}, True
        for a in args:
            if a[:1].upper() not in ALLOWED_LETTERS:
                problems.append(f"line {n}: '{a}' is not allowed")
                ok = False
                continue
            try:
                v[a[0].upper()] = float(a[1:])
            except ValueError:
                problems.append(f"line {n}: cannot read '{a}'")
                ok = False
        if not ok:
            in_run = False
            continue
        if "F" in v:
            if not 300 <= v["F"] <= 12000:
                problems.append(f"line {n}: speed F{v['F']:g} outside 300-12000")
            f_modal = v["F"]
        if "Z" in v:
            if min(abs(v["Z"] - z), abs(v["Z"] - z - ZHOP)) > 0.005:
                problems.append(f"line {n}: Z{v['Z']:g} is neither the layer height {z:.3f} nor lifted {z + ZHOP:.3f}")
            cz = v["Z"]
        nx, ny = v.get("X", x), v.get("Y", y)
        has_xy = "X" in v or "Y" in v
        if has_xy and nx is not None and ny is not None and not (x0 <= nx <= x1 and y0 <= ny <= y1):
            problems.append(f"line {n}: X{nx:g} Y{ny:g} is outside the part's area")
        moved = has_xy and (x is None or y is None or abs(nx - x) > 1e-9 or abs(ny - y) > 1e-9)
        dist = float(np.hypot(nx - x, ny - y)) if moved and x is not None and y is not None else (np.inf if moved else 0.0)
        e = v.get("E")
        kind = None
        if e is not None and e != 0 and not moved:
            kind = "retract"
            if abs(e) > 1.0:
                problems.append(f"line {n}: E{e:g} without moving (retract/prime at most 1 mm)")
            e_still += e
            retracted = max(retracted - e, 0.0) if e > 0 else retracted - e
            in_run = False
        elif e is not None and e < 0 and moved:
            problems.append(f"line {n}: retract while moving")
            in_run = False
        elif e is not None and e > 0 and moved:
            kind = "extrusion"
            if op == "G0":
                problems.append(f"line {n}: G0 must not extrude")
            if abs(cz - z) > 0.005:
                problems.append(f"line {n}: extruding while not at the layer height {z:.3f}")
            elif retracted > 0.05:
                problems.append(f"line {n}: extruding while the filament is still retracted (prime first)")
            if x is not None and y is not None and abs(cz - z) <= 0.005:
                if not in_run:
                    run += 1
                    in_run = True
                segs.append((n, (x, y), (nx, ny), e, run))
                e_total += e
        elif moved:
            kind = "travel"
            in_run = False
            if dist > 1.0:
                if cz is None or cz < z + 0.2:
                    problems.append(f"line {n}: travel of {dist:.1f} mm at layer height (lift first)"
                                    if np.isfinite(dist) else f"line {n}: travel at layer height (lift first)")
                if retracted < RETRACT_MIN:
                    problems.append(f"line {n}: travel without a retract first")
        elif "Z" in v:
            kind = "z"
        if kind and f_modal is not None:
            lo, hi = SPEED_BANDS[kind]
            if not lo <= f_modal <= hi:
                problems.append(f"line {n}: {kind} speed F{f_modal:g} outside {lo}-{hi}")
        x, y = nx, ny

    if abs(e_still) > 0.05:
        problems.append(f"retracts and primes do not balance (net {e_still:+.2f} mm)")
    if want_e and not 0.9 * want_e <= e_total <= 1.1 * want_e:
        problems.append(f"total extrusion {e_total:.2f} mm, expected about {want_e:.2f} mm (all paths printed?)")
    if not jobs:
        return problems

    # ---- which planned path does each extrusion move belong to, and is every path traced?
    bad_rate, off_plan, assigned = 0, 0, []
    printed = [0.0] * len(jobs)
    printed_e = [0.0] * len(jobs)
    flagged_jobs = set()
    for n, a, b, e, rid in segs:
        pa, pb = np.array(a) - (ox, oy), np.array(b) - (ox, oy)
        length = float(np.hypot(*(pb - pa)))
        if length < 0.05:
            assigned.append((n, None, rid))
            continue
        seg = LineString([pa, pb])
        pts = _samples(seg)
        gap = float(np.max(shapely.distance(pts, planned)))
        d_mid = [jl.distance(shapely.Point((pa + pb) / 2)) for jl in job_lines]
        j = int(np.argmin(d_mid))
        if gap > TRACE_TOL_MM:
            off_plan += 1
            if off_plan <= MAX_REPORTED:
                problems.append(f"line {n}: extrusion from ({a[0]:.2f}, {a[1]:.2f}) to ({b[0]:.2f}, {b[1]:.2f}) "
                                f"is {gap:.2f} mm off the planned paths")
            assigned.append((n, None, rid))
            continue
        printed[j] += length
        printed_e[j] += e
        assigned.append((n, j, rid))
        rate = e / length
        # very short segments (curve points) carry a few thousandths of a mm of filament: their
        # error is judged with the rest of the path below, not one by one
        if length >= MIN_SEGMENT_MM and not (1 - RATE_TOL) * expected[j] <= rate <= (1 + RATE_TOL) * expected[j]:
            flagged_jobs.add(j)
            bad_rate += 1
            if bad_rate <= MAX_REPORTED:
                problems.append(f"line {n}: E{e:g} over {length:.3f} mm is {rate:.4f}/mm, expected "
                                f"{expected[j]:.4f}/mm for the {jobs[j][0]} ({jobs[j][2]:.2f} mm wide)")
    if off_plan > MAX_REPORTED:
        problems.append(f"... {off_plan} extrusion moves in total leave the planned paths")
    if bad_rate > MAX_REPORTED:
        problems.append(f"... {bad_rate} extrusion moves in total have the wrong E per mm")

    for j, (feat, path, wd) in enumerate(jobs):
        if j not in flagged_jobs and printed[j] > 1.0:
            rate = printed_e[j] / printed[j]
            if not (1 - PATH_RATE_TOL) * expected[j] <= rate <= (1 + PATH_RATE_TOL) * expected[j]:
                problems.append(f"path {j + 1} ({feat}): E per mm over the whole path is {rate:.4f}, "
                                f"expected {expected[j]:.4f} for {wd:.2f} mm wide")

    # an extrusion run (no travel in between) must stay on one path
    by_run = {}
    for n, j, rid in assigned:
        if j is not None:
            by_run.setdefault(rid, []).append((n, j))
    for rid, items in by_run.items():
        first = items[0][1]
        for n, j in items:
            if j != first:
                problems.append(f"line {n}: extrusion jumps from path {first + 1} ({jobs[first][0]}) to path "
                                f"{j + 1} ({jobs[j][0]}) without a travel")
                break

    extruded = shapely.union_all([LineString([a, b]) for _, a, b, _, _ in
                                  [(n, np.array(a) - (ox, oy), np.array(b) - (ox, oy), e, r) for n, a, b, e, r in segs]]) \
        if segs else None
    for j, (feat, path, wd) in enumerate(jobs):
        line = job_lines[j]
        if extruded is None:
            far = 1.0
        else:
            far = float(np.mean(shapely.distance(_samples(line), extruded) > TRACE_TOL_MM))
        if far > 0.02:
            problems.append(f"path {j + 1} ({feat}, starting at X{path[0][0] + ox:.2f} Y{path[0][1] + oy:.2f}) "
                            f"is not printed ({far:.0%} of it is missing)")
        elif printed[j] > 1.2 * line.length:
            problems.append(f"path {j + 1} ({feat}) is printed more than once "
                            f"({printed[j]:.1f} mm extruded over {line.length:.1f} mm)")

    # paths must start in the given order (reordering is allowed to change nothing else)
    first_seen = []
    for _, j, _ in assigned:
        if j is not None and j not in first_seen:
            first_seen.append(j)
    for a, b in zip(first_seen, first_seen[1:]):
        if b < a:
            problems.append(f"path {b + 1} ({jobs[b][0]}) is printed before path {a + 1} ({jobs[a][0]}); "
                            "keep the given order")
            break
    return problems


# --------------------------------------------------------------------------- the whole file
_Z = re.compile(r"^;\s*(?:Z|Z_HEIGHT)\s*:\s*([\d.]+)", re.I)
_M73L = re.compile(r"\bL(\d+)")
HEATERS = ("M104", "M109", "M140", "M190", "M141", "M191")


def _is_layer_change(line: str) -> bool:
    s = line.lstrip()
    return s.startswith(";") and s[1:].strip().upper().startswith(("LAYER_CHANGE", "CHANGE_LAYER"))


def gcode_file_check(text: str, layers, origin, park_margin_mm: float = 5.0) -> List[str]:
    """Problems in the whole file that would be uploaded (empty = safe). `layers`: the planned
    layers in the file, in order (Layer.z, Layer.region in part-centred mm)."""
    problems: List[str] = []
    lines = text.splitlines()
    marks = [i for i, l in enumerate(lines) if _is_layer_change(l)]
    if not marks:
        return ["the file has no layer changes"]
    first, last = marks[0], marks[-1]

    zs, m73 = [], []
    for k, i in enumerate(marks):
        end = marks[k + 1] if k + 1 < len(marks) else len(lines)
        z = None
        for l in lines[i + 1:min(i + 6, end)]:
            m = _Z.match(l.strip())
            if m:
                z = float(m.group(1))
                break
        if z is None:
            problems.append(f"line {i + 1}: layer change without a Z height")
            continue
        zs.append((i + 1, z))
    for (n0, z0), (n1, z1) in zip(zs, zs[1:]):
        if z1 <= z0 + 1e-6:
            problems.append(f"line {n1}: layer height Z{z1:g} does not rise above the previous layer Z{z0:g}")
    if layers is not None and len(zs) != len(layers):
        problems.append(f"the file has {len(zs)} layers, the plan has {len(layers)}")
    elif layers is not None:
        for (n, z), lay in zip(zs, layers):
            if abs(z - lay.z) > 0.01:
                problems.append(f"line {n}: layer at Z{z:g}, the plan says Z{lay.z:g}")
                break
    if layers is not None and len(layers) > 1:
        h = [b.z - a.z for a, b in zip(layers, layers[1:])]
        for (n0, z0), (n1, z1), step in zip(zs, zs[1:], h):
            if abs((z1 - z0) - step) > 0.01:
                problems.append(f"line {n1}: layer step {z1 - z0:.3f} mm, the plan says {step:.3f} mm")
                break

    for n, raw in enumerate(lines, 1):
        parsed = _parse(raw)
        if parsed is None:
            continue
        code, op, args = parsed
        if op == "M73" and any(a.upper().startswith("L") for a in args):
            m = _M73L.search(code.upper())
            if m:
                m73.append((n, int(m.group(1))))
        if op in ("G29", "M420", "G28") or (op == "G92" and any(a.upper().startswith("Z") for a in args)):
            header_rehome = (op == "G28" and n - 1 < first and [a.upper() for a in args] == ["X", "Y"])
            if not header_rehome:
                problems.append(f"line {n}: '{code}' (homing / bed levelling) is not allowed")
        if op in HEATERS:
            off = any(a.upper() == "S0" for a in args)
            if n - 1 > first and not (off and n - 1 > last):
                problems.append(f"line {n}: '{code}' sets a temperature after the header")
    for (n0, a), (n1, b) in zip(m73, m73[1:]):
        if b != a + 1:
            problems.append(f"line {n1}: M73 L{b} does not follow L{a} (layers must count 1, 2, 3 ...)")
            break

    # footer: after the last layer's extrusion the head leaves the part and the bed drops / head lifts
    parked = _parks(lines, last, layers, origin, park_margin_mm)
    if parked:
        problems.append(parked)
    return problems


def _parks(lines, last, layers, origin, margin) -> Optional[str]:
    if not layers:
        return None
    bx0, by0, bx1, by1 = [], [], [], []
    for l in layers:
        b = l.region.bounds
        if l.region.is_empty:
            continue
        bx0.append(b[0]); by0.append(b[1]); bx1.append(b[2]); by1.append(b[3])
    if not bx0:
        return None
    lo = (min(bx0) + origin[0] - margin, min(by0) + origin[1] - margin)
    hi = (max(bx1) + origin[0] + margin, max(by1) + origin[1] + margin)
    last_extrusion, rel = last, False
    for i in range(last, len(lines)):
        parsed = _parse(lines[i])
        if parsed is None:
            continue
        _, op, args = parsed
        v = {a[0].upper() for a in args}
        if op in ("G0", "G1") and ("E" in v) and ("X" in v or "Y" in v) and \
                any(a.upper().startswith("E") and float(a[1:]) > 0 for a in args):
            last_extrusion = i
    # replay from the last extrusion to find where the head ends up and whether it was lifted
    end_xy, lifted = None, False
    for i in range(last_extrusion + 1, len(lines)):
        parsed = _parse(lines[i])
        if parsed is None:
            continue
        code, op, args = parsed
        v = {a[0].upper(): float(a[1:]) for a in args if re.match(r"[XYZEF][-+]?[\d.]+$", a, re.I)}
        if op == "G91":
            rel = True
        elif op == "G90":
            rel = False
        elif op in ("G0", "G1"):
            if "Z" in v and v["Z"] >= 5:
                lifted = True
            if ("X" in v or "Y" in v) and not rel:
                end_xy = (v.get("X", end_xy[0] if end_xy else None), v.get("Y", end_xy[1] if end_xy else None))
    if end_xy is None or None in end_xy:
        return "the footer does not move the head away from the part"
    inside = lo[0] <= end_xy[0] <= hi[0] and lo[1] <= end_xy[1] <= hi[1]
    if inside:
        return f"the footer leaves the head at X{end_xy[0]:g} Y{end_xy[1]:g}, over the part"
    if not lifted:
        return "the footer does not lift the head or drop the bed before parking"
    return None
