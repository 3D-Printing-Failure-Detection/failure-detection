"""Function first, then executable recovery within explicit action constraints.

Inspired by the notebook's separate function, reachability and cost tools. A
passing continuation needs no speculative margin improvement. The historical
FunctionalPolicy is retained for paired comparison.
"""
import numpy as np

from ..geometry import adapt as A
from ..geometry.ops import boundary_samples, clean, iou, warp
from ..geometry.spec import Verdict, check
from ..types import Layer
from .decide import Decision, DecisionContext
from .functional import FunctionalPolicy, functional_margin
from .recoverability import correction_budget, intervention_tradeoff


class GuardedPolicy(FunctionalPolicy):
    def __init__(self, *args, max_correction_per_layer_mm=.25, **kwargs):
        super().__init__(*args, **kwargs)
        # Validate the explicit engineering assumption before the print starts.
        correction_budget(0, max_correction_per_layer_mm, 0, self.h)
        self.max_correction_per_layer_mm = max_correction_per_layer_mm

    def _targets(self, known):
        k = len(known) - 1
        remaining = self.nominal[k + 1:]
        field = A.Field(self.nominal[k], known[-1])
        points = boundary_samples(self.nominal[k].region, .5)
        correction = float(np.linalg.norm(field(points), axis=1).max()) if len(points) else 0.
        budget = correction_budget(correction, self.max_correction_per_layer_mm, len(remaining), self.h)
        required = budget["required_layers"]
        # Function may permit a permanently deformed shape. Returning to CAD is
        # not required merely because a nominal-return budget was computed.
        yield A.make_plan("hold measured shape", field, remaining, float("inf")), budget
        if required <= 1:
            yield A.Plan("nominal", remaining, 1.), budget
        else:
            # The first future layer takes the first bounded correction step;
            # exactly `required` future layers reach zero correction.
            layers = [Layer(l.z, warp(l.region, lambda p, n=n: field(p, max(0., 1 - n / required))))
                      for n, l in enumerate(remaining, 1)]
            fidelity = float(np.mean([iou(l.region, p.region) for l, p in zip(layers, remaining)]))
            name = "bounded return" if budget["nominal_reachable"] else "bounded partial correction"
            yield A.Plan(name, layers, fidelity), budget
        # Retain slower shape blends, but exclude any faster than the assumed
        # correction rate. This is a field-model bound, not a physical guarantee.
        for decay in (15, 30, 60, 120):
            if decay >= required:
                yield A.make_plan(f"blend over {decay} layers", field, remaining, decay), budget

    def _evaluate(self, known, plan, context, current_plan, budget=None):
        candidate = dict(name=plan.name, plan=plan, predicted=[], executable=True,
                         correction_budget=budget, margin=-1., robust_margin=-1.,
                         uncertainty_mm=self.state.uncertainty_mm)
        try:
            if context.validate_remaining:
                context.validate_remaining(plan.layers)
        except ValueError as exc:
            candidate.update(executable=False, verdict=Verdict(False, [str(exc)]))
            return candidate
        predicted = self.state.predict(plan.layers)
        verdict = check(known + predicted, self.nominal, self.spec, self.h)
        candidate.update(predicted=predicted, verdict=verdict)
        if self.use_fea and self.spec.get("loads"):
            self._stress_filter(known, [candidate])
        if verdict.ok:
            margin = functional_margin(known + predicted, self.nominal, self.spec, self.h, verdict)
            envelope = [margin]
            for sign in (-1, 1):
                future = [Layer(l.z, clean(l.region.buffer(sign * self.state.uncertainty_mm, quad_segs=4)))
                          for l in predicted]
                envelope.append(functional_margin(known + future, self.nominal, self.spec, self.h))
            candidate.update(margin=margin, robust_margin=min(envelope))
        rest = current_plan[len(known):]
        full = sum(l.region.area for l in self.nominal) * self.h
        current_volume = sum(l.region.area for l in rest) * self.h
        command_volume = sum(l.region.area for l in plan.layers) * self.h
        candidate["command_change_fraction"] = sum(a.region.symmetric_difference(b.region).area * self.h
                                                   for a, b in zip(rest, plan.layers)) / max(full, 1e-9)
        candidate["material_tradeoff"] = intervention_tradeoff(current_volume, command_volume, 0., max(full, 1e-9))
        return candidate

    def candidates(self, known, current_plan, context=None):
        context = context or DecisionContext()
        continuation = self._evaluate(known, A.Plan("continue", current_plan[len(known):], 1.),
                                      context, current_plan)
        out = [continuation]
        if continuation["verdict"].ok:
            return out
        if not self.allow_adapt or context.adapts_remaining == 0 or len(known) == len(self.nominal):
            return out
        for target, budget in self._targets(known):
            command = A.Plan("compensate:" + target.name, self.state.compensate(target.layers), target.fidelity)
            out.append(self._evaluate(known, command, context, current_plan, budget))
        return out

    def decide(self, known, current_plan, detection, *, context=None):
        if detection.label == "detach":
            return Decision("scrap", "part detached: nothing to build on")
        context = context or DecisionContext()
        candidates = self.candidates(known, current_plan, context)
        considered = [{k: v for k, v in c.items() if k not in ("plan", "predicted", "verdict", "stress")}
                      | dict(ok=c["verdict"].ok, violations=c["verdict"].violations,
                             fidelity=c["plan"].fidelity, adapts_remaining=context.adapts_remaining)
                      for c in candidates]
        feasible = [c for c in candidates if c["verdict"].ok and c["executable"]]
        continuation = next((c for c in feasible if c["name"] == "continue"), None)
        if continuation:
            return Decision("continue", "continuation passes function checks; no intervention justified",
                            continuation["plan"], continuation["verdict"], considered)
        if not feasible:
            reason = "continuation fails and no executable recovery meets the spec"
            if context.adapts_remaining == 0:
                reason += "; adaptation budget exhausted"
            return Decision("scrap", reason, None, candidates[0]["verdict"], considered)
        chosen = max(feasible, key=lambda c: (c["robust_margin"], c["margin"],
                                              -c["command_change_fraction"], c["plan"].fidelity))
        return Decision("adapt", "continuation fails; choose a feasible correction within the action limits",
                        chosen["plan"], chosen["verdict"], considered)
