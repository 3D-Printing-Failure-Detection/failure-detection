"""The feedback loop. Runs against any printer with the SimPrinter interface."""
from dataclasses import dataclass, field
from typing import List, Optional

from .. import camera, config as C
from ..geometry.spec import Verdict, check
from ..types import Layer


@dataclass
class Event:
    layer: int
    label: str
    deviation_mm: float
    action: str
    reason: str
    measured: Optional[Layer] = None
    considered: list = field(default_factory=list)
    image: object = None                 # camera frame that raised the alarm
    expected: Optional[Layer] = None     # what the plan said this layer should be
    plan_layers: Optional[List[Layer]] = None   # adapted layers chosen (if action == adapt)


@dataclass
class Result:
    outcome: str                      # finished | scrapped
    final: Optional[Verdict]
    events: List[Event]
    layers_printed: int
    filament_mm: float                # filament used by this attempt (incl. wasted part of a scrap)
    functional: bool


def filament_of(layers: List[Layer], h: float) -> float:
    return sum(l.region.area for l in layers) * h / C.FILAMENT_AREA


class Controller:
    def __init__(self, printer, segmenter, detector, policy, spec, nominal, layer_h,
                 quiet_horizon: Optional[int] = None, max_adapts=3, log=None):
        self.pr, self.seg, self.det, self.pol = printer, segmenter, detector, policy
        self.spec, self.nominal, self.h = spec, nominal, layer_h
        self.horizon, self.max_adapts = quiet_horizon, max_adapts
        self.log = log or (lambda *a: None)

    def run(self) -> Result:
        pr, events, adapts, quiet = self.pr, [], 0, 0
        known = list(pr.printed)                          # layers believed printed (clean prefix)
        while pr.n_done < pr.n_total and not pr.cancelled:
            pr.print_layer()
            img, exp = pr.camera_image(), pr.expected_layer()
            meas = self.seg.measure(img, camera.expected_mask(exp))
            if hasattr(pr, "record_measurement"):
                pr.record_measurement(Layer(exp.z, meas))
            det = self.det.detect(img, exp.region, meas)
            if det.label == "ok":
                known.append(exp)
                quiet += 1
                if self.horizon and quiet >= self.horizon:
                    pr.fast_forward()
                    break
                continue
            quiet = 0
            pr.pause()
            measured = Layer(exp.z, meas)
            known.append(measured)
            self.log(f"layer {pr.n_done-1}: {det.label} dev={det.deviation_mm:.1f}mm")
            dec = self.pol.decide(known, pr.plan, det)
            if dec.action == "adapt" and adapts >= self.max_adapts:
                dec.action, dec.reason = "scrap", "too many re-plans on one print"
            events.append(Event(pr.n_done - 1, det.label, det.deviation_mm, dec.action, dec.reason,
                                measured, dec.considered, img, exp,
                                dec.plan.layers if dec.action == "adapt" else None))
            self.log(f"  -> {dec.action}: {dec.reason}")
            if dec.action == "scrap":
                pr.cancel()
                break
            if dec.action == "adapt":
                pr.replace_remaining(dec.plan.layers)
                adapts += 1
            else:
                pr.resume()
        fil = filament_of(pr.printed, self.h)
        if pr.cancelled:
            return Result("scrapped", None, events, pr.n_done, fil, False)
        final = check(pr.printed, self.nominal, self.spec, self.h)
        return Result("finished", final, events, pr.n_done, fil, final.ok)
