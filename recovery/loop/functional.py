"""Opt-in recovery using measured state, predicted deposition and function margin.

Hard feasibility is still geometry.spec.check (and optional FEA). Candidate
uncertainty envelopes inform ranking; they are not probability guarantees.
"""
import numpy as np

from ..geometry import adapt as A, features as F
from ..geometry.ops import clean
from ..geometry.spec import check, MAX_DEFECT_LAYERS, OVERHANG_AREA
from ..models.classical import Detection
from ..types import Layer
from .decide import Decision, RulePolicy
from .observation import ObservationState


def functional_margin(layers, nominal, spec, layer_h, verdict=None):
    """Minimum normalized constraint slack; CAD similarity does not enter it."""
    verdict = verdict or check(layers, nominal, spec, layer_h)
    if not verdict.ok:
        return -1.0
    metrics = verdict.metrics
    margins = [1 - metrics["max_overhang_mm2"] / OVERHANG_AREA]

    def interval(value, bounds):
        lo, hi = bounds
        return min(value - lo, hi - value) / max((hi - lo) / 2, 1e-9)

    if spec.get("container"):
        margins.append(interval(metrics["capacity_mm3"], spec["container"]["capacity_mm3"]))
    if spec.get("material_volume_mm3"):
        margins.append(interval(metrics["material_mm3"], spec["material_volume_mm3"]))
    for hole in spec.get("holes", []):
        prefix = f"hole_{hole['center']}"
        margins.extend([interval(metrics[prefix + "_d"], hole["diameter"]),
                        1 - metrics[prefix + "_offset_mm"] / max(hole["tol_center"], 1e-9)])
    if spec.get("min_wall_mm"):
        t = spec["min_wall_mm"]
        # Reserve above the required wall, while accounting for nominal corners.
        reserve = 0.
        for extra in (.25, .5):
            thin = sum(F.thin_frac(l, t + extra) - F.thin_frac(p, t + extra) > .03
                       for l, p in zip(layers, nominal))
            if thin <= MAX_DEFECT_LAYERS:
                reserve = extra / .5
        margins.extend([reserve, 1 - metrics["thin_layers"] / (MAX_DEFECT_LAYERS + 1)])
    for rule in spec.get("present", []):
        lo, hi = rule["z"]
        for layer, plan in zip(layers, nominal):
            if lo <= layer.z <= hi and plan.region.area:
                floor = rule["min_area_frac"]
                margins.append((layer.region.area / plan.region.area - floor) / max(1 - floor, 1e-9))
    return float(min(margins))


class FunctionalPolicy(RulePolicy):
    def __init__(self, *args, predict=True, rank_margin=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = ObservationState()
        self.predict, self.rank_margin = predict, rank_margin
        self.previous = None

    def observe(self, expected, measured, detection, index):
        estimate = self.state.observe(expected, measured)
        nominal = self.nominal[index]
        warning = self._function_warning(estimate, nominal)
        self.previous = (estimate, nominal)
        if warning and detection.label == "ok":
            detection = Detection("functional_risk", detection.deviation_mm,
                                  dict(detection.info, reason=warning,
                                       uncertainty_mm=self.state.uncertainty_mm))
        return estimate, detection

    def _function_warning(self, layer, nominal):
        f, pf = F.base(layer), F.base(nominal)
        c = self.spec.get("container")
        if c:
            if layer.z <= c["base_h"]:
                if f["area"] < .85 * pf["area"] or f["n_main_holes"]:
                    return "base lost material or opened a hole"
            elif f["hole_area"] < .5 * pf["hole_area"]:
                return "container wall opened or cavity blocked"
        t = self.spec.get("min_wall_mm")
        if t and F.thin_frac(layer, t + .25) - F.thin_frac(nominal, t + .25) > .03:
            return "wall approaching minimum thickness"
        for hole in self.spec.get("holes", []):
            holes = F.hole_near(layer, hole["center"], hole["diameter"][1])
            if not holes:
                return "required hole blocked"
            (x, y), diameter = F.inscribed(max(holes, key=lambda p: p.area))
            offset = np.hypot(x - hole["center"][0], y - hole["center"][1])
            lo, hi = hole["diameter"]
            if diameter < lo + .25 or diameter > hi - .25 or offset > hole["tol_center"] - .125:
                return "hole fit approaching its limit"
        for rule in self.spec.get("connected", []):
            if rule["z"][0] <= layer.z <= rule["z"][1] and f["n"] != rule.get("components", 1):
                return "required connection broken"
        for rule in self.spec.get("present", []):
            if rule["z"][0] <= layer.z <= rule["z"][1] and f["area"] < rule["min_area_frac"] * pf["area"]:
                return "required feature losing material"
        if self.previous:
            prev, pprev = self.previous
            if F.excess_overhang(layer, prev, nominal, pprev) > .8 * OVERHANG_AREA:
                return "support approaching its limit"
        return None

    def candidates(self, known, current_plan):
        if not self.predict:
            candidates = super().candidates(known, current_plan)
            if self.rank_margin:
                for candidate in candidates:
                    candidate["margin"] = functional_margin(known + candidate["plan"].layers,
                        self.nominal, self.spec, self.h, candidate["verdict"])
                    candidate["robust_margin"] = candidate["margin"]
            return candidates
        k = len(known) - 1
        rest = current_plan[k + 1:]
        commands = [A.Plan("continue", rest, 1.)]
        if self.allow_adapt and rest:
            targets = [A.Plan("nominal", self.nominal[k + 1:], 1.)]
            targets += A.candidate_plans(self.nominal[k], known[-1], self.nominal[k + 1:],
                                         relief=self._relief())
            commands += [A.Plan("compensate:" + p.name, self.state.compensate(p.layers), p.fidelity)
                         for p in targets]
        out = []
        for plan in commands:
            predicted = self.state.predict(plan.layers)
            verdict = check(known + predicted, self.nominal, self.spec, self.h)
            candidate = dict(name=plan.name, plan=plan, predicted=predicted, verdict=verdict)
            if self.rank_margin:
                candidate["margin"] = functional_margin(known + predicted, self.nominal, self.spec, self.h, verdict)
                # Material may be thinner or thicker than the point forecast.
                # Keep the measured prefix fixed: compensation cannot repair it.
                uncertainty = self.state.uncertainty_mm
                envelope = []
                for sign in (-1, 1):
                    future = [Layer(l.z, clean(l.region.buffer(sign * uncertainty, quad_segs=4))) for l in predicted]
                    v = check(known + future, self.nominal, self.spec, self.h)
                    envelope.append(functional_margin(known + future, self.nominal, self.spec, self.h, v))
                candidate["robust_margin"] = min([candidate["margin"], *envelope])
                candidate["uncertainty_mm"] = self.state.uncertainty_mm
            out.append(candidate)
        if self.use_fea and self.spec.get("loads"):
            self._stress_filter(known, out)
        return out

    def pick(self, candidates):
        if not self.rank_margin:
            return super().pick(candidates)
        feasible = [c for c in candidates if c["verdict"].ok]
        considered = [dict(name=c["name"], ok=c["verdict"].ok, fidelity=c["plan"].fidelity,
                           violations=c["verdict"].violations, margin=c.get("margin"),
                           robust_margin=c.get("robust_margin"), uncertainty_mm=c.get("uncertainty_mm"))
                      for c in candidates]
        if not feasible:
            decision = super().pick(candidates)
            decision.considered = considered
            return decision
        chosen = max(feasible, key=lambda c: (c.get("robust_margin", -1), c.get("margin", -1),
                                              c["name"] == "continue", c["plan"].fidelity))
        # Treat tiny score differences as ties to avoid repeated toolpath changes
        # driven by camera noise. Never retain a failing continuation.
        continuation = next((c for c in feasible if c["name"] == "continue"), None)
        if continuation and (chosen.get("robust_margin", -1) - continuation.get("robust_margin", -1) <= .02
                             and chosen.get("margin", -1) - continuation.get("margin", -1) <= .02):
            chosen = continuation
        action = "continue" if chosen["name"] == "continue" else "adapt"
        return Decision(action, f"{chosen['name']} selected by predicted functional margin (0.02 tie tolerance)",
                        chosen["plan"], chosen["verdict"], considered)
