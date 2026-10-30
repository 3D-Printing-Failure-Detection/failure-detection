"""Stage 3+4: assess whether the part can still meet its spec and decide
continue / adapt / scrap. Rules first; an LLM planner can pick among the rule-verified
candidates (loop/llm.py) but can never pick one that fails the spec check."""
from dataclasses import dataclass, field
from typing import List, Optional

from shapely.ops import unary_union

from ..geometry import adapt as A
from ..geometry import features as F
from ..geometry.spec import Verdict, check
from ..types import Layer


@dataclass
class Decision:
    action: str                                  # continue | adapt | scrap
    reason: str
    plan: Optional[A.Plan] = None
    verdict: Optional[Verdict] = None
    considered: List[dict] = field(default_factory=list)


class RulePolicy:
    def __init__(self, spec, nominal: List[Layer], layer_h, allow_adapt=True,
                 cheap_scrap_frac=0.03, cheap_scrap_dev=3.0, use_fea=False, fea_h=1.5):
        self.spec, self.nominal, self.h = spec, nominal, layer_h
        self.allow_adapt = allow_adapt
        self.use_fea, self.fea_h = use_fea, fea_h      # also stress-test candidates that pass the geometry checks
        self.cheap_frac, self.cheap_dev = cheap_scrap_frac, cheap_scrap_dev

    def _relief(self):
        """Through-holes named in the spec: keep their nominal footprint open."""
        holes = self.spec.get("holes")
        if not holes:
            return None

        def fn(layer):
            ps = [h for hs in holes for h in F.hole_near(layer, hs["center"], hs["diameter"][1])]
            return unary_union(ps) if ps else None
        return fn

    def candidates(self, known: List[Layer], current_plan: List[Layer]):
        """Evaluate every option against the spec: -> list of dict(name, plan, verdict)."""
        k = len(known) - 1
        rest = current_plan[k + 1:]
        out = []
        cont = A.Plan("continue", rest, 1.0)
        out.append(dict(name="continue", plan=cont,
                        verdict=check(known + rest, self.nominal, self.spec, self.h)))
        if self.allow_adapt and rest:
            for p in A.candidate_plans(current_plan[k], known[-1], rest, relief=self._relief()):
                out.append(dict(name=p.name, plan=p,
                                verdict=check(known + p.layers, self.nominal, self.spec, self.h)))
        if self.use_fea and self.spec.get("loads"):
            self._stress_filter(known, out)
        return out

    def _stress_filter(self, known, cands):
        """Geometry says a candidate is fine; now simulate the loads on the finished part.
        Check every eligible candidate: optional policies may select any passing option."""
        from ..fea import stress_test
        for c in cands:
            v = c["verdict"]
            if not v.ok:
                continue
            rep = stress_test(known + c.get("predicted", c["plan"].layers), self.spec, self.h, self.fea_h)
            c["stress"] = rep
            if not rep.ok:
                v.ok = False
                v.violations += rep.violations

    def decide(self, known: List[Layer], current_plan: List[Layer], detection) -> Decision:
        k = len(known) - 1
        if detection.label == "detach":
            return Decision("scrap", "part detached: nothing to build on")
        if k / len(self.nominal) < self.cheap_frac and detection.deviation_mm > self.cheap_dev:
            return Decision("scrap", "severe failure very early: restarting is cheaper")
        cands = self.candidates(known, current_plan)
        return self.pick(cands)

    def pick(self, cands) -> Decision:
        considered = [dict(name=c["name"], ok=c["verdict"].ok, fidelity=c["plan"].fidelity,
                           violations=c["verdict"].violations) for c in cands]
        for c in cands:                           # continue first, then adapt by fidelity
            if c["verdict"].ok:
                act = "continue" if c["name"] == "continue" else "adapt"
                return Decision(act, f"{c['name']} keeps the part within spec",
                                c["plan"], c["verdict"], considered)
        worst = cands[0]["verdict"]
        return Decision("scrap", "no option meets the spec: " + "; ".join(worst.violations[:2]),
                        None, worst, considered)


class AlertOnlyPolicy(RulePolicy):
    """Baseline (c): pause and alert a human who can only continue or scrap."""
    def __init__(self, *a, **kw):
        super().__init__(*a, allow_adapt=False, cheap_scrap_frac=0.0, **kw)
