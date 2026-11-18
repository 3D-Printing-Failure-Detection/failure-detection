"""What the agent can see and do on a printer. Works on BambuPrinter (real P1S) and SimPrinter
alike, because both share the controller's printer interface. Nothing here decides anything."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..geometry import features as F
from ..geometry.ops import iou
from ..geometry.spec import OVERHANG_AREA, check
from ..loop.controller import filament_of
from ..types import Layer

# spec keys that describe one checkable feature each (see geometry/spec.py)
FEATURE_KEYS = ("container", "min_wall_mm", "material_volume_mm3", "holes", "connected", "present")
LISTED = ("holes", "connected", "present")
SUPPORT = "layer not supported"          # prefix of the checker's overhang violation


@dataclass
class Observation:
    layer: int                       # index of the layer that just finished
    z: float
    label: str                       # what it looks like: ok, collapse, warp, shift, under_extrusion, detach
    flagged_px: int = 0              # size of the area that differs from the plan (in the model's crop)
    type_prob: dict = field(default_factory=dict)
    image: object = field(repr=False, default=None)
    expected: Optional[Layer] = field(repr=False, default=None)


class PrinterTools:
    def __init__(self, printer, part, watcher=None, yaw_deg=0.0):
        """watcher: models.p1s_layer.LayerWatcher (or anything with the same `check`), needed only for
        the per-layer camera check. yaw_deg: the part's rotation on the plate (0 = as sliced)."""
        self.pr, self.part, self.watcher, self.yaw = printer, part, watcher, yaw_deg
        self.nominal = part.layers()
        self._prev_frame = None
        self.known: List[Layer] = list(getattr(printer, "printed", []))[:printer.n_done]

    # ---- sense ----
    def status(self) -> dict:
        link = getattr(self.pr, "link", None)
        s = dict(layer=self.pr.n_done, total=self.pr.n_total,
                 paused=bool(getattr(self.pr, "paused", False)), cancelled=bool(self.pr.cancelled))
        if link is not None:
            s.update(state=link.get("gcode_state"), remaining_min=link.get("mc_remaining_time"))
        return s

    def next_layer(self):
        """Block until the next layer has finished printing. Returns "paused" instead if someone
        else paused the printer (dent tests: from the screen, the app, or a planned pause)."""
        step = self.pr.print_layer()
        self._sync()
        return step if step == "paused" else None

    def _sync(self):
        """Believe the plan for layers that finished without being looked at."""
        while len(self.known) < self.pr.n_done:
            self.known.append(self.pr.plan[len(self.known)])

    def is_paused(self) -> bool:
        """Is the printer paused right now (someone may have resumed it on the screen)?"""
        link = getattr(self.pr, "link", None)
        if link is not None:
            return link.get("gcode_state") == "PAUSE"
        return bool(getattr(self.pr, "paused", False))

    def snapshot(self):
        """The printer camera's frame, as the camera sees it (BGR)."""
        return self.pr.snapshot()

    def observe(self) -> Observation:
        """The frame after the layer that just finished, compared with the frame after the layer before
        and with the plan, in the camera's own view. The camera cannot measure a shape from this angle,
        so the belief about the part stays the plan; the observation says whether, and how, the print
        looks different from it."""
        plan = self.pr.plan[:self.pr.n_done]
        try:
            frame = self.snapshot()
            prev = frame if self._prev_frame is None else self._prev_frame
            origin = getattr(self.pr, "origin", None) or self.part.origin
            r = self.watcher.check(prev[..., ::-1], frame[..., ::-1], plan, origin, self.yaw)
        except Exception as e:           # no frame, or the part is not where the plan says: say so, go on
            return Observation(self.pr.n_done - 1, plan[-1].z, "unchecked", 0, {"error": f"{type(e).__name__}: {e}"},
                               None, plan[-1])
        self._prev_frame = frame
        return Observation(self.pr.n_done - 1, plan[-1].z, r["kind"], int(r["flagged_px"]),
                           r.get("type_prob", {}), frame, plan[-1])

    def believe_top(self, layers: List[Layer]):
        """Record measured shapes for the top len(layers) printed layers (e.g. after a hand dent)."""
        self.known[len(self.known) - len(layers):] = list(layers)

    # ---- check ----
    def _support(self, stack, start):
        """Only layers still to be printed can hang over air; printed layers exist as they are."""
        worst = 0.0
        for j in range(max(start, 1), len(stack)):
            worst = max(worst, F.excess_overhang(stack[j], stack[j - 1], self.nominal[j], self.nominal[j - 1]))
        ok = worst <= OVERHANG_AREA
        return dict(feature="new layers supported by the layer below", ok=ok,
                    detail="" if ok else f"{worst:.0f} mm2 of a new layer would hang over air "
                                         f"(limit {OVERHANG_AREA:.0f})", metrics=dict(max_overhang_mm2=round(worst, 2)))

    def function_report(self, current: Optional[Layer] = None, future: Optional[List[Layer]] = None) -> dict:
        """Does the finished part still meet its spec if the remaining layers are `future` (default:
        the current plan)? One verdict per feature (each hole, the container, walls...) so a cosmetic
        change is never confused with a lost function. Uses the existing checker one rule at a time.
        `current`: assume this (measured) shape for the layer that just finished."""
        known = self.known if current is None else self.known[:-1] + [current]
        rest = self.pr.plan[len(known):] if future is None else list(future)
        stack = known + rest
        nominal, h = self.nominal, self.part.layer_h
        features = [self._support(stack, len(known))]
        for key in FEATURE_KEYS:
            rule = self.part.spec.get(key)
            if not rule:
                continue
            for i, item in enumerate(rule if key in LISTED else [rule]):
                v = check(stack, nominal, {key: [item] if key in LISTED else item}, h)
                own = [x for x in v.violations if not x.startswith(SUPPORT)]
                name = f"{key}[{i}]" if key in LISTED else key
                if key == "holes":
                    name = f"hole at {item['center']} (dia {item['diameter'][0]}-{item['diameter'][1]} mm)"
                if key == "container":
                    name = "holds liquid (closed walls, solid base, capacity)"
                features.append(dict(feature=name, ok=not own, detail="; ".join(own),
                                     metrics={k: round(float(m), 3) for k, m in v.metrics.items()
                                              if k != "max_overhang_mm2"}))
        return dict(continue_ok=all(f["ok"] for f in features), features=features)

    def try_correction(self, fade_layers) -> Optional[dict]:
        """Build and check ONE correction plan: the remaining layers start on the measured top layer
        and return to the original shape over `fade_layers` layers (None or inf = keep the measured
        shape to the top). Boundary-role-aware fit (Ishaan's ContourResidual: outer edges matched to
        outer edges, inner to inner), so a thin wall is not mis-fitted. Returns the plan whether or
        not it passes (`report["continue_ok"]`), with its fidelity to the design (mean IoU)."""
        from ..loop.contour import ContourResidual, _warp_rings
        k = len(self.known)
        rest = self.pr.plan[k:]
        if not rest:
            return None
        fade = float("inf") if fade_layers is None else float(fade_layers)
        res = ContourResidual(self.pr.plan[k - 1], self.known[-1])
        layers = []
        for n, lay in enumerate(rest):
            g = 1.0 if not np.isfinite(fade) else max(0.0, 1.0 - n / fade)
            layers.append(lay if g <= 1e-3 else
                          Layer(lay.z, _warp_rings(lay.region, lambda pts, interior, g=g: g * res(pts, interior))))
        report = self.function_report(future=layers)
        fidelity = float(np.mean([iou(a.region, b.region) for a, b in zip(layers, self.nominal[k:])]))
        return dict(fade_layers=None if not np.isfinite(fade) else int(fade), fidelity=round(fidelity, 4),
                    layers=layers, report=report)

    def correction(self, fades=(10, 15, 30, float("inf"))) -> Optional[dict]:
        """The automatic search: try each fade length, return the passing plan closest to the
        original design, or None."""
        best = None
        for fade in fades:
            fix = self.try_correction(fade)
            if fix is None:
                return None
            if fix["report"]["continue_ok"] and (best is None or fix["fidelity"] > best["fidelity"]):
                best = fix
        return best

    def cost(self) -> dict:
        h = self.part.layer_h
        used = filament_of(self.pr.plan[:self.pr.n_done], h)
        total = filament_of(self.pr.plan, h)
        return dict(filament_used_mm=round(used), filament_total_mm=round(total),
                    fraction_done=round(self.pr.n_done / max(self.pr.n_total, 1), 3),
                    scrap_wastes_mm=round(used), restart_needs_mm=round(total))

    # ---- act (only ever called through the gate) ----
    def pause(self):
        self.pr.pause()

    def resume(self):
        self.pr.resume()

    def stop(self):
        self.pr.cancel()

    def correction_gcode(self, layers: List[Layer]) -> str:
        """The G-code that `correct` would send (the simulator writes it the same way)."""
        own = getattr(self.pr, "correction_gcode", None)
        if own:
            return own(layers)
        from ..printer.bambu import resume_gcode
        return resume_gcode(layers, self.part.layer_h, self.pr.n_done, self.part.origin)

    def correct(self, layers: List[Layer]):
        """Replace the remaining layers (real P1S: stop, upload new G-code, start; unproven)."""
        self.pr.replace_remaining(layers)
