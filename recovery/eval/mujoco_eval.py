"""Paired full-print transfer evaluation across MuJoCo printer rigs.

The same classical perception and rule policy are frozen for every profile.
Every monitored layer is imaged; no clean-prefix or quiet-horizon shortcuts.
Raw rows distinguish successful in-place prints from a hypothetical clean restart.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..deform import KINDS, random_failure
from ..geometry.ops import iou
from ..geometry.spec import check
from ..loop.controller import Controller, filament_of
from ..loop.decide import AlertOnlyPolicy, RulePolicy
from ..loop.functional import FunctionalPolicy
from ..models.classical import ClassicalDetector, ClassicalSegmenter
from ..parts import PART_NAMES, get_part
from ..printer.profiles import PROFILES
from .run_eval import METHODS, ScrapPolicy

EXPERIMENT_METHODS = METHODS + ["measured", "predictive", "functional"]


class ObservedSegmenter(ClassicalSegmenter):
    """Evaluation-only ground truth access; never changes a measurement/decision."""
    def __init__(self, printer):
        self.printer = printer
        self.ious = []

    def measure(self, image, expected_mask=None):
        measured = super().measure(image, expected_mask)
        self.ious.append(iou(measured, self.printer.printed[-1].region))
        return measured


def run_case(job):
    from ..printer.mujoco import MujocoPrinter
    profile, part_name, kind, seed, methods, clear_on_adapt = job
    part = get_part(part_name)
    nominal = part.layers()
    nominal_check = check(nominal, nominal, part.spec, part.layer_h)
    if not nominal_check.ok:
        raise ValueError(f"nominal {part_name} fails: {nominal_check}")
    failure = None if kind == "clean" else random_failure(nominal, np.random.default_rng(seed), kind=kind)
    full = filament_of(nominal, part.layer_h)
    rows = []
    for method in methods:
        started = time.perf_counter()
        with MujocoPrinter(nominal, failure, seed, profile, clear_on_adapt,
                            render=method != "ignore") as pr:
            seg = ObservedSegmenter(pr)
            events = []
            if method == "ignore":
                pr.fast_forward()
                verdict = check(pr.printed, nominal, part.spec, part.layer_h)
                functional, outcome = verdict.ok, "finished"
            else:
                if method in ("measured", "predictive", "functional"):
                    policy = FunctionalPolicy(part.spec, nominal, part.layer_h,
                                              predict=method != "measured", rank_margin=method == "functional")
                else:
                    policy = {"ours": RulePolicy, "alert": AlertOnlyPolicy, "scrap": ScrapPolicy}[method](
                        part.spec, nominal, part.layer_h)
                res = Controller(pr, seg, ClassicalDetector(), policy, part.spec,
                                 nominal, part.layer_h).run()
                functional, outcome, verdict, events = res.functional, res.outcome, res.final, res.events
            used = filament_of(pr.printed, part.layer_h)
            k0 = failure.k0 if failure else len(nominal)
            post_failure = [e for e in events if e.layer >= k0]
            rows.append(dict(
                profile=profile, part=part_name, kind=kind, seed=seed, method=method,
                clear_failure_on_adapt=clear_on_adapt, k0=k0 if failure else -1,
                severity=float(failure.p["severity"]) if failure else 0,
                outcome=outcome, in_place=functional, scrapped=pr.cancelled,
                defective_finished=(outcome == "finished" and not functional),
                # A restart is a cost assumption, never counted as observed success.
                filament_ratio=used / full,
                filament_with_assumed_clean_restart=(used + (full if pr.cancelled else 0)) / full,
                n_layers=len(nominal), layers_printed=pr.n_done, frames=pr.camera_frames,
                n_events=len(events), pre_failure_alarms=sum(e.layer < k0 for e in events),
                alarm_delay=post_failure[0].layer-k0 if post_failure else -1,
                first_label=events[0].label if events else "",
                adaptations=len(pr.replacements), measurement_iou=float(np.mean(seg.ious)) if seg.ious else None,
                tracking_rms_mm=float(pr.tracking_rms_mm), max_tracking_mm=pr.max_tracking_mm,
                executed_waypoints=pr.executed_waypoints,
                violations=verdict.violations if verdict else [events[-1].reason],
                actions=[dict(layer=e.layer, label=e.label, action=e.action, reason=e.reason,
                              considered=e.considered) for e in events],
                seconds=time.perf_counter()-started))
    return rows


def summarize(rows):
    summary = []
    ignored = {(r["profile"], r["part"], r["kind"], r["seed"]): r for r in rows if r["method"] == "ignore"}
    for profile in sorted({r["profile"] for r in rows}):
        for method in EXPERIMENT_METHODS:
            for split in ("clean", "failure"):
                selected = [r for r in rows if r["profile"] == profile and r["method"] == method
                            and (r["kind"] == "clean") == (split == "clean")]
                if not selected:
                    continue
                n = len(selected)
                harmful = [r for r in selected if (b := ignored.get((profile, r["part"], r["kind"], r["seed"])))
                           and not b["in_place"]]
                summary.append(dict(profile=profile, method=method, split=split, n=n,
                    in_place=sum(r["in_place"] for r in selected)/n,
                    scrapped=sum(r["scrapped"] for r in selected)/n,
                    defective_finished=sum(r["defective_finished"] for r in selected)/n,
                    adapted=sum(r["adaptations"] > 0 for r in selected)/n,
                    n_harmful=len(harmful),
                    recovered_harmful=sum(r["in_place"] for r in harmful)/len(harmful) if harmful else None,
                    mean_filament_with_assumed_restart=float(np.mean([r["filament_with_assumed_clean_restart"] for r in selected])),
                    mean_measurement_iou=float(np.mean([r["measurement_iou"] for r in selected])) if method != "ignore" else None))
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparison(rows):
    """Count actual in-place rescues and regressions on identical scenarios."""
    key = lambda r: (r["profile"], r["part"], r["kind"], r["seed"])
    by_method = {m: {key(r): r for r in rows if r["method"] == m} for m in EXPERIMENT_METHODS}
    comparisons = []
    for baseline, treatment in (("ignore", "ours"), ("ignore", "measured"), ("ignore", "predictive"),
                                ("ignore", "functional"), ("ours", "measured"), ("measured", "predictive"),
                                ("predictive", "functional"), ("ours", "functional")):
        for split in ("clean", "failure"):
            keys = sorted(k for k in by_method[baseline].keys() & by_method[treatment].keys()
                          if (k[2] == "clean") == (split == "clean"))
            pairs = [(by_method[baseline][k], by_method[treatment][k]) for k in keys]
            if not pairs:
                continue
            comparisons.append(dict(baseline=baseline, treatment=treatment, split=split, n=len(pairs),
                baseline_in_place=sum(a["in_place"] for a, b in pairs),
                treatment_in_place=sum(b["in_place"] for a, b in pairs),
                gained=[list(key(b)) for a, b in pairs if b["in_place"] and not a["in_place"]],
                lost=[list(key(b)) for a, b in pairs if a["in_place"] and not b["in_place"]]))
    return comparisons


def source_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(list((root / "recovery").rglob("*.py")) + list((root / "parts").glob("*.yaml"))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--parts", nargs="+", default=PART_NAMES)
    parser.add_argument("--kinds", nargs="+", choices=["clean"] + KINDS, default=["clean"] + KINDS)
    parser.add_argument("--methods", nargs="+", choices=EXPERIMENT_METHODS, default=METHODS)
    parser.add_argument("--n", type=int, default=2, help="seeds per part/failure/profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clear-failure-on-adapt", action="store_true",
                        help="optimistic ablation: adaptation removes the prescribed failure cause")
    parser.add_argument("--out", default="results/mujoco")
    args = parser.parse_args(argv)
    if args.n < 1 or args.workers < 1:
        parser.error("--n and --workers must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runs.jsonl").exists():
        parser.error("output already contains runs.jsonl; use a fresh directory")
    jobs = [(p, part, kind, seed, args.methods, args.clear_failure_on_adapt)
            for p in args.profiles for part in args.parts for kind in args.kinds
            for seed in range(args.seed, args.seed + args.n)]
    manifest = dict(arguments=vars(args), profiles={k: asdict(PROFILES[k]) for k in args.profiles},
                    python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "shapely", "opencv-python-headless", "mujoco")},
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=source_digest(), expected_runs=len(jobs)*len(args.methods),
                    status="running", assumptions=["quasi-static waypoints", "geometric bead deposition",
                    "0.1 mm bead-fusion closing before failure injection",
                    "prescribed failures", "controlled current-layer illumination", "exact camera calibration",
                    "geometry-only function verdicts", "restart cost is hypothetical"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, errors = [], []
    # Spawn isolates GL contexts; forking an initialized EGL runtime is unsafe.
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(run_case, job): job for job in jobs}
        with (output / "runs.jsonl").open("w") as stream:
            for future in as_completed(futures):
                try:
                    batch = future.result()
                except Exception as exc:
                    errors.append(dict(job=futures[future], error=repr(exc)))
                    print(f"ERROR {futures[future][:4]}: {exc}", flush=True)
                    continue
                rows.extend(batch)
                for row in batch:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{len(rows)}/{manifest['expected_runs']} runs: {futures[future][:4]}", flush=True)
    rows.sort(key=lambda r: (r["profile"], r["part"], r["kind"], r["seed"], r["method"]))
    write_csv(output / "runs.csv", rows)
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "paired.json").write_text(json.dumps(paired_comparison(rows), indent=2))
    manifest.update(status="failed" if errors else "complete", completed_runs=len(rows), errors=errors,
                    final_source_sha256=source_digest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} scenarios failed; see manifest.json")


if __name__ == "__main__":
    main()
