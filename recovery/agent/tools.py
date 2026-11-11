"""What the agent can see and do on a printer. Works on BambuPrinter (real P1S) and SimPrinter
alike, because both share the controller's printer interface. Nothing here decides anything."""
from dataclasses import dataclass, field
from typing import List, Optional

from .. import camera
from ..geometry.spec import check
from ..loop.controller import filament_of
from ..types import Layer

# spec keys that describe one checkable feature each (see geometry/spec.py)
FEATURE_KEYS = ("container", "min_wall_mm", "material_volume_mm3", "holes", "connected", "present")


@dataclass
class Observation:
    layer: int                       # index of the layer that just finished
    z: float
    label: str                       # detector label: ok, collapse, warp, shift, under_extrusion, detach
    deviation_mm: float
    image: object = field(repr=False, default=None)
    expected: Optional[Layer] = field(repr=False, default=None)
    measured: Optional[Layer] = field(repr=False, default=None)


class PrinterTools:
    def __init__(self, printer, part, segmenter, detector):
        self.pr, self.part, self.seg, self.det = printer, part, segmenter, detector
        self.nominal = part.layers()
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
        """Block until the next layer has finished printing."""
        self.pr.print_layer()

    def observe(self) -> Observation:
        """Camera frame of the layer that just finished, measured against the plan. Updates the
        agent's belief of what is printed: nominal where it looked fine, measured where it did not."""
        img, exp = self.pr.camera_image(), self.pr.expected_layer()
        meas = self.seg.measure(img, camera.expected_mask(exp))
        d = self.det.detect(img, exp.region, meas)
        measured = Layer(exp.z, meas)
        self.known.append(exp if d.label == "ok" else measured)
        return Observation(self.pr.n_done - 1, exp.z, d.label, float(d.deviation_mm), img, exp, measured)

    def function_report(self, current: Optional[Layer] = None) -> dict:
        """If we simply continue: does the finished part still meet its spec? One verdict per
        feature (each hole, the container, walls...) so a cosmetic change is never confused with
        a lost function. Uses the existing checker one rule at a time. `current`: assume this
        (measured) shape for the layer that just finished instead of the agent's belief."""
        known = self.known if current is None else self.known[:-1] + [current]
        stack = known + self.pr.plan[len(known):]
        nominal, h = self.nominal, self.part.layer_h
        support = check(stack, nominal, {}, h)          # overhang rule only
        features = [dict(feature="layers supported by the layer below", ok=support.ok,
                         detail="; ".join(support.violations))]
        for key in FEATURE_KEYS:
            rule = self.part.spec.get(key)
            if not rule:
                continue
            for i, item in enumerate(rule if key in ("holes", "connected", "present") else [rule]):
                sub = {key: [item] if key in ("holes", "connected", "present") else item}
                v = check(stack, nominal, sub, h)
                own = [x for x in v.violations if x not in support.violations]
                name = key if key not in ("holes", "connected", "present") else f"{key}[{i}]"
                if key == "holes":
                    name = f"hole at {item['center']} (dia {item['diameter'][0]}-{item['diameter'][1]} mm)"
                features.append(dict(feature=name, ok=not own, detail="; ".join(own),
                                     metrics={k: round(float(m), 3) for k, m in v.metrics.items()}))
        return dict(continue_ok=all(f["ok"] for f in features), features=features)

    def cost(self) -> dict:
        h = self.part.layer_h
        used = filament_of(self.pr.plan[:self.pr.n_done], h)
        total = filament_of(self.pr.plan, h)
        return dict(filament_used_mm=round(used), filament_total_mm=round(total),
                    fraction_done=round(self.pr.n_done / max(self.pr.n_total, 1), 3),
                    scrap_wastes_mm=round(used), restart_needs_mm=round(total))

    def believe_measured(self, obs: Observation):
        """Record the measured shape (not the plan) for the layer that just finished."""
        self.known[-1] = obs.measured

    # ---- act (only ever called through the gate) ----
    def pause(self):
        self.pr.pause()

    def resume(self):
        self.pr.resume()

    def stop(self):
        self.pr.cancel()
