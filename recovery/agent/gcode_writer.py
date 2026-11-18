"""The language model writes the G-code of the correction layers (`--plan-fix gcode`).

Only the layers that differ from the design are written by the model, one layer per call, all
layers at once in parallel. For the 3 mm tray dent that is 10 layers (31-40); everything above
is the original design and is copied from the sliced Bambu Studio file when there is one (the
simulator has none, so our writer stands in).

  per layer   the model gets the layer's height, its toolpaths (outline loops and gap fills, in bed
              coordinates, from the planned shape), line widths, the extrusion formula, speeds and
              the travel rules, and writes the G-code: Z moves, travels, retracts, every E value.
  check       a safety check (allowed commands only, Z only at the layer height or lifted, moves
              inside the part's area, E per mm close to the formula, retracts balanced), then the
              layer is read back with our G-code parser and compared with the planned shape.
  retry       a failing layer goes back to the model with the problems, up to `rounds` times; then
              our own writer writes that layer, and the log says so.
  whole part  the layers as read back are checked with Sri's function checks (holds water, every
              layer supported); if that fails, our writer's file is used instead.
  file        resume header + the model's layers + the original's later layers + footer.

The checks are recovery/geometry/gcode_check.py (per layer, and for the whole uploaded file).
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import shapely

from .. import config as C
from .. import gcode as G
from ..geometry.gcode_check import gcode_check, gcode_file_check
from ..geometry.ops import iou
from ..types import Layer
from .deciders import parse_reply
from .usage import labelled

RETRACT, ZHOP = 0.8, 0.4
FEED, TRAVEL, ZFEED, RFEED = 1800, 7200, 600, 1800
SIMPLIFY_MM = 0.02           # straight runs become 2 points; the dent's curve keeps its points

SYSTEM = f"""You write the G-code for ONE layer of a 3D print on a Bambu Lab P1S (0.4 mm nozzle, PLA,
1.75 mm filament). The printer is already set up: absolute X/Y/Z (G90), RELATIVE extrusion (M83:
each E is the filament length for that move only), nozzle hot, fan on. Write only motion lines.

For EACH path in `paths`, in the given order, write exactly:
  G1 E-{RETRACT} F{RFEED}                 retract
  G1 Z<z + {ZHOP}> F{ZFEED}               lift clear of the part
  ;TYPE:<the path's feature>
  G0 X<first point x> Y<first point y> F{TRAVEL}   travel to the path's first point
  G1 Z<z> F{ZFEED}                        back down to the layer height
  G1 E{RETRACT} F{RFEED}                  prime
  then for every next point: G1 X<x> Y<y> E<e> F{FEED}
      where e = (distance from the previous point in mm) * e_per_mm of that path, 5 decimals.
Use the coordinates exactly as given (3 decimals). Never use any other command (no G28, G29, G2,
G3, G90, G91, G92, M-codes, temperatures). Never extrude while lifted, never move Z anywhere else.
Reply with JSON only: {{"gcode": "<the lines, separated by \\n>"}}"""

ALLOWED_LETTERS = set("XYZEF")


def _bed(path, origin):
    return [[round(float(x) + origin[0], 3), round(float(y) + origin[1], 3)] for x, y in path]


def layer_jobs(region, layer_idx, w=C.LINE_W):
    """The toolpaths our writer would print for this region: (feature, polyline, width), the same
    loops and gap fills as recovery.gcode.write_layers(gap_fill=True), lightly simplified."""
    tagged = G.layer_paths(region, layer_idx, w, 2, tagged=True)
    jobs = [(feat, path, w) for feat, path in tagged]
    jobs += [("Gap infill", path, gw) for path, gw in G.gap_fill_paths(region, [p for _, p in tagged], w)]
    out = []
    for feat, path, width in jobs:
        line = shapely.LineString(path)
        if line.length < 0.2:
            continue
        out.append((feat, np.array(line.simplify(SIMPLIFY_MM).coords), float(width)))
    return out


def layer_context(layer: Layer, layer_no: int, layer_h, origin, w=C.LINE_W):
    jobs = layer_jobs(layer.region, layer_no - 1, w)
    paths = [dict(feature=f, width_mm=round(wd, 3), e_per_mm=round(G._e(1.0, wd, layer_h), 5), points=_bed(p, origin))
             for f, p, wd in jobs]
    return jobs, dict(layer=layer_no, z=round(layer.z, 3), layer_height_mm=layer_h,
                      e_formula="e_per_mm = line width * layer height / (pi * (1.75/2)^2)", paths=paths)


def markers(layer: Layer, layer_no: int, layer_h, w=C.LINE_W):
    return f";LAYER_CHANGE\n;Z:{layer.z:.3f}\n;HEIGHT:{layer_h:.3f}\n;WIDTH:{w:.3f}\nM73 L{layer_no}\n"


def _strip(text):
    text = re.sub(r"^```[a-zA-Z]*\n|```\s*$", "", text.strip(), flags=re.M)
    return text.strip() + "\n"


def check_layer(text, layer: Layer, jobs, layer_h, origin, margin_mm=2.0) -> List[str]:
    """Safety problems in one layer of G-code (empty = safe to print): recovery.geometry.gcode_check."""
    return gcode_check(text, layer, jobs, layer_h, origin, margin_mm)


def read_back(text, layer: Layer, layer_no, layer_h, origin) -> Layer:
    """The layer as our parser reads the G-code (same parser as for Bambu Studio files)."""
    got = G.gcode_to_layers("M83\n" + markers(layer, layer_no, layer_h) + text, origin=origin)
    return Layer(layer.z, got[0].region if got else shapely.Polygon())       # keep the planned height exactly


def original_layers(text: Optional[str], start_idx: int) -> Optional[str]:
    """The sliced file's layers from index `start_idx` (0-based) to the last one, cut before the
    machine end G-code (our footer parks the head). None if this cannot be done safely."""
    if not text:
        return None
    lines = text.splitlines()
    codes = [l.split(";", 1)[0].strip().upper() for l in lines]
    if any(c.startswith("M82") for c in codes) and not any(c.startswith("M83") for c in codes):
        return None                                     # absolute extrusion: E values would not fit
    idx = [i for i, l in enumerate(lines) if G.is_layer_change(l)]
    if start_idx >= len(idx):
        return None
    end = next((i for i in range(idx[-1], len(lines))
                if any(m in lines[i] for m in ("MACHINE_END_GCODE_START", "EXECUTABLE_BLOCK_END"))), None)
    if end is None:
        return None
    block = codes[idx[start_idx]:end]
    if any(c.split()[0] in ("G28", "G29") for c in block if c) or any(c.startswith("G29") for c in block):
        return None
    return "\n".join(lines[idx[start_idx]:end]) + "\n"


class LLMGcodeWriter:
    """The model writes each changed layer's G-code; checks accept it or send it back."""

    def __init__(self, complete, name="llm", rounds=3, max_layers=15, workers=10):
        self.complete, self.name = labelled(complete, "write G-code"), name
        self.rounds, self.max_layers, self.workers = rounds, max_layers, workers
        self.transcript = []

    def _one(self, layer, layer_no, layer_h, origin):
        jobs, ctx = layer_context(layer, layer_no, layer_h, origin)
        ours, _ = G.write_layers([layer], layer_h, start_idx=layer_no - 1, origin=origin, gap_fill=True,
                                 retract_mm=RETRACT, zhop_mm=ZHOP, first_retract=True)
        ours = ours.split(f";WIDTH:{C.LINE_W:.3f}\n", 1)[-1]            # body only (markers added here)
        tries = []
        for _ in range(self.rounds):
            user = json.dumps(dict(ctx, previous_attempts=[dict(problems=t["problems"][:10]) for t in tries]))
            rec = dict(layer=layer_no, attempt=len(tries) + 1)
            try:
                reply = self.complete(SYSTEM, user)
                text = _strip(str(parse_reply(reply)["gcode"]))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                tries.append(dict(problems=[f"no usable reply ({type(e).__name__})"]))
                self.transcript.append(rec)
                continue
            problems = check_layer(text, layer, jobs, layer_h, origin)
            back = read_back(text, layer, layer_no, layer_h, origin) if not problems else None
            if back is not None:
                match = iou(back.region, layer.region) if not back.region.is_empty else 0.0
                rec["shape_match"] = round(match, 4)
                if match < 0.95:
                    problems.append(f"the printed shape matches the planned layer only {match:.0%} (need 95%)")
            rec.update(problems=problems, lines=text.count("\n"))
            self.transcript.append(rec)
            tries.append(dict(problems=problems))
            if not problems:
                return dict(layer=layer_no, text=text, by=self.name, attempts=len(tries), shape=back)
        return dict(layer=layer_no, text=ours, by="our writer", attempts=len(tries),
                    shape=read_back(ours, layer, layer_no, layer_h, origin),
                    why=tries[-1]["problems"][:3] if tries else [])

    def write(self, tools, fix) -> dict:
        """Adds to `fix`: gcode (the whole file), layers (the model's layers as read back, then the
        unchanged ones), gcode_by (who wrote each layer)."""
        k, layer_h, origin = tools.pr.n_done, tools.part.layer_h, tools.part.origin
        plan_rest = tools.pr.plan[k:]
        layers = fix["layers"]
        n_new = next((j for j in range(len(layers)) if all(a is b for a, b in zip(layers[j:], plan_rest[j:]))),
                     len(layers))
        if n_new == 0 or n_new > self.max_layers:
            fix["gcode_by"] = dict(summary=f"our writer ({n_new} changed layers; the model writes 1-{self.max_layers})")
            return fix
        with ThreadPoolExecutor(min(self.workers, n_new)) as ex:
            done = list(ex.map(lambda j: self._one(layers[j], k + j + 1, layer_h, origin), range(n_new)))
        new_shapes = [d["shape"] for d in done]
        report = tools.function_report(future=new_shapes + list(layers[n_new:]))
        by_model = sum(d["by"] == self.name for d in done)
        info = dict(model_layers=by_model, our_layers=n_new - by_model, changed_layers=n_new,
                    attempts=sum(d["attempts"] for d in done),
                    per_layer=[dict(layer=d["layer"], by=d["by"], attempts=d["attempts"], **({"why": d["why"]} if "why" in d else {}))
                               for d in done])
        if not report["continue_ok"]:
            info["summary"] = ("our writer: the model's layers as read back fail the function check ("
                               + "; ".join(f["detail"] for f in report["features"] if not f["ok"]) + ")")
            fix["gcode_by"] = info
            return fix
        from ..printer.bambu import FOOTER, resume_header
        pr = tools.pr
        head = resume_header(layers[0].z, k, getattr(pr, "nozzle", 220), getattr(pr, "bed", 55),
                             getattr(pr, "rehome", ""), ZHOP)
        body = "".join(markers(layers[j], k + j + 1, layer_h) + done[j]["text"] for j in range(n_new))
        tail = original_layers(getattr(pr, "original_gcode", None) or getattr(tools.part, "gcode_text", None), k + n_new)
        tail_by = "copied from the sliced file"
        if tail is not None:                     # the copied layers must start at the next planned height
            first = G.gcode_to_layers("M83\n" + tail, origin=origin)
            if not first or abs(first[0].z - layers[n_new].z) > 0.01:
                tail = None
        if tail is None:
            tail, _ = G.write_layers(list(layers[n_new:]), layer_h, start_idx=k + n_new, origin=origin, bambu=True,
                                     gap_fill=True, retract_mm=RETRACT, zhop_mm=ZHOP, first_retract=True)
            tail_by = "our writer (no usable sliced file)"
        info.update(rest_layers=len(layers) - n_new, rest_by=tail_by,
                    summary=f"{by_model}/{n_new} changed layers written by {self.name}"
                            + (f", {n_new - by_model} by our writer" if by_model < n_new else "")
                            + f"; the other {len(layers) - n_new} layers {tail_by}")
        full = head + body + "; --- original layers ---\n" + tail + FOOTER
        problems = gcode_file_check(full, list(new_shapes) + list(layers[n_new:]), origin)
        if problems:
            info["summary"] = "our writer: the assembled file failed the file check (" + "; ".join(problems[:3]) + ")"
            info["file_problems"] = problems
            fix["gcode_by"] = info
            return fix
        fix.update(gcode=full,
                   layers=new_shapes + list(layers[n_new:]), report=report, gcode_by=info)
        return fix

    @property
    def full_transcript(self):
        return self.transcript
