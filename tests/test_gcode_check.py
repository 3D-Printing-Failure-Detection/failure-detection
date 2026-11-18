"""gcode_check / gcode_file_check: the safety check for G-code written by a language model."""
import re

import pytest

from recovery import gcode as G
from recovery.agent.gcode_writer import RETRACT, ZHOP, layer_context
from recovery.geometry.gcode_check import gcode_check, gcode_file_check
from recovery.parts import get_part
from recovery.printer.bambu import resume_gcode
from tests.test_agent import _protocol_gcode

PART = get_part("parts/print/water-tray/water-tray.yaml")
LAYER = PART.layers()[40]
JOBS, CTX = layer_context(LAYER, 41, PART.layer_h, PART.origin)
GOOD = _protocol_gcode(CTX)
Z = CTX["z"]


def check(text):
    return gcode_check(text, LAYER, JOBS, PART.layer_h, PART.origin)


def blocks(text=GOOD):
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith(f"G1 E-{RETRACT}") and cur:
            out.append(cur)
            cur = []
        cur.append(line)
    out.append(cur)
    return out


def join(bs):
    return "\n".join(line for b in bs for line in b)


def says(problems, fragment):
    return any(fragment in p for p in problems)


# ---------------------------------------------------------------- accepts
def test_accepts_the_protocol_reply_and_our_own_writer():
    assert check(GOOD) == []
    ours, _ = G.write_layers([LAYER], PART.layer_h, start_idx=40, origin=PART.origin, gap_fill=True,
                             retract_mm=RETRACT, zhop_mm=ZHOP, first_retract=True)
    body = ours.split(f";WIDTH:{0.42:.3f}\n", 1)[-1]
    assert check(body) == []


def test_every_problem_names_a_line_or_path():
    for text in (GOOD + "\nG28", GOOD + "\nG1 X500 Y500 F7200", join(blocks()[:-1])):
        assert all(re.search(r"line \d+|path \d+|total extrusion|balance", p) for p in check(text))


# ---------------------------------------------------------------- the stand-in's rules, kept
@pytest.mark.parametrize("line, says_", [("G28", "not allowed"), ("M104 S0", "not allowed"),
                                          (f"G1 Z{Z - 1:.3f}", "neither the layer height"),
                                          ("G1 X500 Y500 F7200", "outside the part"), ("G1 E3 F1800", "at most 1 mm"),
                                          ("G1 F20", "outside 300-12000"), ("G1 X10 Y10 A5", "not allowed")])
def test_basic_rules_still_hold(line, says_):
    assert says(check(GOOD + "\n" + line), says_), line


def test_lifted_and_never_lowered_and_doubled_flow():
    assert says(check(GOOD.replace(f"G1 Z{Z:.3f} F600", f"G1 Z{Z + 0.4:.3f} F600")), "while not at the layer height")
    doubled = "\n".join(l.replace(" E0.0", " E0.1") for l in GOOD.splitlines())
    assert says(check(doubled), "E per mm") or says(check(doubled), "total extrusion")


# ---------------------------------------------------------------- what the stand-in missed
def test_a_dropped_path_is_named():
    bs = blocks()
    for drop in (bs[-1], min(bs, key=len)):
        p = check(join([b for b in bs if b is not drop]))
        assert says(p, "is not printed"), p


def test_extra_extrusion_off_the_plan():
    first = CTX["paths"][0]
    x, y = first["points"][0]
    extra = f"\nG1 X{x + 6:.3f} Y{y:.3f} E{6 * first['e_per_mm']:.5f} F1800"
    assert says(check(GOOD + extra), "off the planned paths")


def test_wrong_flow_on_one_path_is_judged_against_that_path():
    bs = blocks()
    scaled = [re.sub(r" E([\d.]+) F1800$", lambda m: f" E{float(m.group(1)) * 1.3:.5f} F1800", l)
              if l.startswith("G1 X") else l for l in bs[6]]       # one thin gap-fill path at 130 % flow
    bs[6] = scaled
    assert says(check(join(bs)), "E per mm") or says(check(join(bs)), "expected")


def test_travel_must_be_lifted_and_follow_a_retract_and_be_primed():
    bs = blocks()
    no_lift = [[l for l in b if not l.startswith("G1 Z")] for b in bs[:2]] + bs[2:]
    no_lift[1] = [l for l in bs[1] if not l.startswith("G1 Z")]
    assert says(check(join([bs[0]] + [no_lift[1]] + bs[2:])), "lift first") or \
        says(check(join([bs[0]] + [no_lift[1]] + bs[2:])), "not at the layer height")
    no_retract = [bs[0], [l for l in bs[1] if not l.startswith(f"G1 E-{RETRACT}")]] + bs[2:]
    assert says(check(join(no_retract)), "without a retract")
    no_prime = [bs[0], [l for l in bs[1] if l != f"G1 E{RETRACT} F1800"]] + bs[2:]
    assert says(check(join(no_prime)), "still retracted") or says(check(join(no_prime)), "balance")


def test_extrusion_may_not_jump_inside_a_path_or_between_paths():
    pts = CTX["paths"][0]["points"]
    L = GOOD.splitlines()
    i = next(i for i, l in enumerate(L) if l.startswith("G1 X"))
    jump = f"G1 X{pts[0][0] + 1:.3f} Y{pts[0][1] + 8:.3f} E0.2 F1800"
    assert check("\n".join(L[:i + 1] + [jump] + L[i + 1:]))
    two = blocks()
    merged = two[0] + [l for l in two[1] if l.startswith("G1 X")]        # path 2 printed straight after path 1
    assert says(check(join([merged] + two[2:])), "jumps from path 1") or check(join([merged] + two[2:]))


def test_paths_in_the_given_order_and_once():
    assert says(check(join(blocks()[::-1])), "keep the given order")
    assert says(check(GOOD + "\n" + join([blocks()[0]])), "more than once") or \
        says(check(GOOD + "\n" + join([blocks()[0]])), "total extrusion")


@pytest.mark.parametrize("old, new, kind", [(" F1800", " F12000", "extrusion speed"),
                                            (" F7200", " F600", "travel speed"),
                                            (f"G1 Z{Z:.3f} F600", f"G1 Z{Z:.3f} F9000", "z speed")])
def test_speeds_by_move_type(old, new, kind):
    text = GOOD.replace(old, new) if kind != "z speed" else GOOD.replace(f"G1 Z{Z:.3f} F600", new)
    assert says(check(text), kind), kind


# ---------------------------------------------------------------- the whole file
LAYERS = PART.layers()[30:]
FILE = resume_gcode(LAYERS, PART.layer_h, 30, PART.origin)


def file_check(text):
    return gcode_file_check(text, LAYERS, PART.origin)


def test_the_file_our_own_writer_makes_passes():
    assert file_check(FILE) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda t: t.replace(";Z:6.400", ";Z:6.200", 1), "does not rise"),
    (lambda t: t.replace("M73 L33", "M73 L35", 1), "M73 L35"),
    (lambda t: t.replace(";LAYER_CHANGE", "G28\n;LAYER_CHANGE", 1), "homing"),
    (lambda t: t + "G29\n", "homing"),
    (lambda t: t.replace("M73 L40\n", "M73 L40\nM104 S250\n", 1), "temperature after the header"),
    (lambda t: t.rsplit("G1 X65 Y245 F12000", 1)[0] + "; end\n", "footer"),
    (lambda t: t.replace("G1 X65 Y245 F12000\nG1 Y265 F3000", "G1 X113 Y138 F12000"), "over the part"),
    (lambda t: t.replace(";LAYER_CHANGE", ";NOTHING"), "no layer changes"),
])
def test_file_problems(mutate, fragment):
    assert says(file_check(mutate(FILE)), fragment), fragment


def test_a_header_xy_rehome_is_the_one_allowed_g28():
    rehomed = resume_gcode(LAYERS, PART.layer_h, 30, PART.origin, rehome_gcode="lift_xy")
    assert "G28 X Y" in rehomed and file_check(rehomed) == []
    assert says(file_check(rehomed.replace("G28 X Y", "G28")), "homing")


# ---------------------------------------------------------------- the fallback writer retracts before its first travel
def test_a_layer_written_on_its_own_retracts_before_its_first_travel():
    on_its_own, _ = G.write_layers([LAYER], PART.layer_h, start_idx=40, origin=PART.origin, gap_fill=True,
                                   retract_mm=RETRACT, zhop_mm=ZHOP, first_retract=True)
    lines = [l for l in on_its_own.splitlines() if not l.startswith(";")]
    assert lines[0] == f"G1 E-{RETRACT:.2f} F1800"
    plain, _ = G.write_layers([LAYER], PART.layer_h, start_idx=40, origin=PART.origin, gap_fill=True,
                              retract_mm=RETRACT, zhop_mm=ZHOP)
    assert not [l for l in plain.splitlines() if not l.startswith(";")][0].startswith("G1 E-")   # default unchanged


def test_tiny_segments_are_judged_with_their_path_not_one_by_one():
    # live finding: a 0.08 mm curve segment with an E off by 20 % is 0.0006 mm^3 of filament
    lines = GOOD.splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("G1 X") and "E0.0" in l)
    x, y = [float(re.search(rf"{a}([\d.]+)", lines[i]).group(1)) for a in "XY"]
    tiny = lines[:i + 1] + [f"G1 X{x + 0.08:.3f} Y{y:.3f} E0.00231 F1800"]
    assert not any("0.082" in p or "0.08" in p and "expected" in p for p in check("\n".join(tiny + lines[i + 1:])))
    # ... but a whole path 12 % over is caught
    bs = blocks()
    over = [re.sub(r" E([\d.]+) F1800$", lambda m: f" E{float(m.group(1)) * 1.12:.5f} F1800", l)
            if l.startswith("G1 X") else l for l in bs[-1]]
    assert says(check(join(bs[:-1] + [over])), "E per mm")
