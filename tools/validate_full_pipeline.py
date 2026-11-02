"""Exercise authored specs, FEA gates, RGB recovery and replacement G-code.

The default case explicitly uses the optimistic transient-cause assumption.
The main benchmark reports persistent causes separately.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from recovery.deform import random_failure
from recovery.fea import stress_test
from recovery.gcode import gcode_to_layers
from recovery.geometry.ops import iou
from recovery.loop.controller import Controller
from recovery.loop.decide import RulePolicy
from recovery.models.classical import ClassicalDetector, ClassicalSegmenter
from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.printer.profiles import PROFILES
from recovery.specgen import author_spec


class AuditedPolicy(RulePolicy):
    stress_checks = 0

    def candidates(self, known, current_plan):
        candidates = super().candidates(known, current_plan)
        self.stress_checks += sum("stress" in candidate for candidate in candidates)
        return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part = get_part("cup")
    nominal = part.layers()
    authored = author_spec(nominal, part.layer_h, use_llm=False, h=2)
    if not authored["verified"]:
        raise RuntimeError(authored["issues"])
    spec = authored["spec"]
    pre = stress_test(nominal, spec, part.layer_h, 2)
    if not pre.ok:
        raise RuntimeError(pre.violations)
    rows = []
    for profile in PROFILES:
        failure = random_failure(nominal, np.random.default_rng(10), kind="collapse")
        policy = AuditedPolicy(spec, nominal, part.layer_h, use_fea=True, fea_h=2)
        with MujocoPrinter(nominal, failure, 10, profile, clear_failure_on_adapt=True) as pr:
            result = Controller(pr, ClassicalSegmenter(), ClassicalDetector(), policy,
                                spec, nominal, part.layer_h).run()
            final = stress_test(pr.printed, spec, part.layer_h, 2) if result.outcome == "finished" else None
            adapted = [event for event in result.events if event.action == "adapt"]
            roundtrips = []
            for i, (gc, event) in enumerate(zip(pr.replacements, adapted)):
                # write_layers returns a body, used under M83 by real resume files.
                gc = "G21\nG90\nM83\n" + gc
                path = out / f"{profile}-replacement-{i}.gcode"
                path.write_text(gc)
                parsed = gcode_to_layers(gc, origin=tuple(np.array(pr.profile.bed_mm)/2))
                roundtrips.append(dict(layers=len(parsed), expected_layers=len(event.plan_layers),
                    min_iou=min(iou(a.region, b.region) for a, b in zip(parsed, event.plan_layers))))
            rows.append(dict(profile=profile, authored_spec_verified=authored["verified"],
                preprint_fea_ok=pre.ok, candidate_stress_checks=policy.stress_checks,
                outcome=result.outcome, functional=result.functional,
                geometry_violations=result.final.violations if result.final else ["scrapped"],
                final_fea_ok=final.ok if final else None, frames=pr.camera_frames,
                adaptations=len(pr.replacements), gcode_roundtrips=roundtrips,
                clear_failure_on_adapt=True))
            print(rows[-1], flush=True)
            (out / "results.json").write_text(json.dumps(rows, indent=2))
    if not all(row["functional"] and row["final_fea_ok"] and row["candidate_stress_checks"] > 0
               and row["adaptations"] > 0 and all(r["layers"] == r["expected_layers"] and r["min_iou"] > .95
               for r in row["gcode_roundtrips"]) for row in rows):
        raise SystemExit("full-pipeline check did not satisfy every gate; see results.json")


if __name__ == "__main__":
    main()
