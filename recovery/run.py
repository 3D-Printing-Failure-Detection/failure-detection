"""One-call scenario runner shared by the CLI and the Gradio app."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .deform import Failure, random_failure
from .gcode import write_gcode
from .geometry.spec import Verdict, check
from .loop.controller import Controller, Result
from .loop.decide import AlertOnlyPolicy, RulePolicy
from .models.classical import ClassicalDetector, ClassicalSegmenter
from .parts import Part, get_part
from .sim import SimPrinter
from .types import Layer


@dataclass
class Scenario:
    part: Part
    nominal: List[Layer]
    failure: Failure
    result: Optional[Result]
    printed: List[Layer]
    ignore_verdict: Verdict
    gcode: str


def make_models(kind="classical", ckpt_dir="runs"):
    if kind == "torch":
        from .models.nets import NeuralDetector, NeuralSegmenter
        return NeuralSegmenter(ckpt_dir), NeuralDetector(ckpt_dir)
    return ClassicalSegmenter(), ClassicalDetector()


def run_scenario(part_name, kind="collapse", seed=0, severity=None, policy="ours",
                 models="classical", log=None, from_start=False, use_fea=False) -> Scenario:
    part = get_part(part_name)
    nom = part.layers()
    rng = np.random.default_rng(seed)
    f = random_failure(nom, rng, kind=kind, severity=severity)
    seg, det = make_models(models)
    pol_cls = AlertOnlyPolicy if policy == "alert" else RulePolicy
    pol = pol_cls(part.spec, nom, part.layer_h, use_fea=use_fea)
    # layers before the failure are clean; skip imaging them unless asked (full-print run)
    pr = SimPrinter(nom, f, seed=seed, start_idx=0 if from_start else f.k0)
    res = Controller(pr, seg, det, pol, part.spec, nom, part.layer_h,
                     log=log).run()
    ig = SimPrinter(nom, f, seed=seed, start_idx=f.k0)
    ig.fast_forward()
    gcode, _ = write_gcode(pr.printed if not pr.cancelled else nom[:pr.n_done], part.layer_h,
                           origin=part.origin)
    return Scenario(part, nom, f, res, pr.printed, check(ig.printed, nom, part.spec, part.layer_h), gcode)
