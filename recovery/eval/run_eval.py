"""Simulation evaluation: inject failures at random heights, run the loop and the three
baselines, check the final geometry against the part's function spec.

  ignore   keep printing (no monitoring)
  scrap    scrap and restart on any alarm
  alert    pause and alert a human who can only continue or scrap
  ours     detect -> measure -> assess -> adapt (or continue / scrap)

A scrapped print is restarted from scratch, which succeeds, but costs the filament already
used. So every method ends 'functional' or 'defective'; they differ in how much filament they
waste and how often they have to restart.
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..deform import KINDS, random_failure
from ..geometry.spec import check
from ..loop.controller import Controller, filament_of
from ..loop.decide import AlertOnlyPolicy, Decision, RulePolicy
from ..models.classical import ClassicalDetector, ClassicalSegmenter
from ..parts import PART_NAMES, get_part
from ..sim import SimPrinter

METHODS = ["ignore", "scrap", "alert", "ours"]


class ScrapPolicy(RulePolicy):
    def decide(self, known, current_plan, detection):
        return Decision("scrap", "baseline: scrap on any alarm")


def run_one(args):
    part_name, kind, seed, methods = args
    part = get_part(part_name)
    nom = part.layers()
    rng = np.random.default_rng(seed)
    f = random_failure(nom, rng, kind=kind)
    full = filament_of(nom, part.layer_h)
    rows = []
    for m in methods:
        t0 = time.time()
        pr = SimPrinter(nom, f, seed=seed, start_idx=f.k0)
        base = dict(part=part_name, kind=kind, seed=seed, method=m, k0=f.k0,
                    severity=round(f.p.get("severity", 0), 2), n_layers=len(nom))
        if m == "ignore":
            pr.fast_forward()
            v = check(pr.printed, nom, part.spec, part.layer_h)
            rows.append(dict(base, outcome="finished", in_place=v.ok, restarted=False,
                             functional=v.ok, filament_ratio=1.0, first_label="", alarm_delay=-1,
                             adapted=False, violations="; ".join(v.violations), secs=time.time() - t0))
            continue
        pol_cls = {"scrap": ScrapPolicy, "alert": AlertOnlyPolicy, "ours": RulePolicy}[m]
        pol = pol_cls(part.spec, nom, part.layer_h)
        res = Controller(pr, ClassicalSegmenter(), ClassicalDetector(), pol, part.spec, nom,
                         part.layer_h, quiet_horizon=25).run()
        restarted = res.outcome == "scrapped"
        used = res.filament_mm + (full if restarted else 0.0)
        first = res.events[0] if res.events else None
        rows.append(dict(
            base, outcome=res.outcome, in_place=(res.functional and not restarted),
            restarted=restarted, functional=(res.functional or restarted),
            filament_ratio=used / full, first_label=first.label if first else "",
            alarm_delay=(first.layer - f.k0) if first else -1,
            adapted=any(e.action == "adapt" for e in res.events),
            violations="; ".join(res.final.violations) if res.final else "scrapped",
            secs=time.time() - t0))
    return rows


def summarise(rows):
    out = []
    harmful = {(x["part"], x["kind"], x["seed"]) for x in rows
               if x["method"] == "ignore" and not x["functional"]}
    for m in METHODS:
        r = [x for x in rows if x["method"] == m]
        if not r:
            continue
        n = len(r)
        h = [x for x in r if (x["part"], x["kind"], x["seed"]) in harmful]
        out.append(dict(
            method=m, n=n, n_harmful=len(h),
            recovered_when_harmful=(sum(x["in_place"] for x in h) / len(h)) if h else float("nan"),
            functional_in_place=sum(x["in_place"] for x in r) / n,
            restarted=sum(x["restarted"] for x in r) / n,
            defective=sum(not x["functional"] for x in r) / n,
            filament_vs_clean_print=float(np.mean([x["filament_ratio"] for x in r])),
        ))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=PART_NAMES)
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--n", type=int, default=6, help="scenarios per (part, kind)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    jobs = [(p, k, a.seed * 1000 + i, METHODS) for p in a.parts for k in a.kinds for i in range(a.n)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_one, jobs) for r in rs]
    Path(a.out).mkdir(exist_ok=True)
    with open(Path(a.out) / "eval_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    summ = summarise(rows)
    print(f"{len(jobs)} scenarios x {len(METHODS)} methods in {time.time()-t0:.0f}s\n")
    print(f"{'method':8s} {'ok in place':>12s} {'restarted':>10s} {'defective':>10s} "
          f"{'filament x clean':>17s} {'saved when harmful':>19s}")
    for s in summ:
        print(f"{s['method']:8s} {s['functional_in_place']:12.0%} {s['restarted']:10.0%} "
              f"{s['defective']:10.0%} {s['filament_vs_clean_print']:17.2f} "
              f"{s['recovered_when_harmful']:19.0%}")
    print(f"('saved when harmful' = share of the {summ[0]['n_harmful']} failures that ruin an ignored print "
          "which still end as a functional part with no restart)")
    with open(Path(a.out) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader(), w.writerows(summ)
    print("\nby failure kind (ours): in place / restarted / defective")
    for k in a.kinds:
        r = [x for x in rows if x["method"] == "ours" and x["kind"] == k]
        if r:
            n = len(r)
            print(f"  {k:16s} {sum(x['in_place'] for x in r)/n:5.0%} {sum(x['restarted'] for x in r)/n:5.0%} "
                  f"{sum(not x['functional'] for x in r)/n:5.0%}")
    return rows


if __name__ == "__main__":
    main()
