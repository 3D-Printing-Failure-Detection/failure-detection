"""Audit completed paired policy runs and export small, reviewable evidence.

python tools/audit_functional_experiment.py DIR [DIR ...] --out docs/validation/functional
Optional --baseline DIR checks rerun ignore/ours outcomes against the earlier study.
"""
import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.eval.mujoco_eval import paired_comparison, source_digest


KEY = ("profile", "part", "kind", "seed", "method")
METHODS = ("ignore", "ours", "measured", "predictive", "functional", "guarded", "contour")


def read_rows(path):
    return [json.loads(line) for line in (path / "runs.jsonl").read_text().splitlines()]


def identity(row):
    return tuple(row[k] for k in KEY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digest = source_digest()
    old = {identity(r): r for r in read_rows(args.baseline)} if args.baseline else {}
    audit, summary, compact, changed = [], [], [], {}
    for directory in args.directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = read_rows(directory)
        assert manifest["status"] == "complete" and not manifest["errors"]
        assert manifest["source_sha256"] == manifest["final_source_sha256"] == digest
        assert len(rows) == manifest["expected_runs"] == manifest["completed_runs"]
        assert len({identity(r) for r in rows}) == len(rows)
        assert not manifest["arguments"]["clear_failure_on_adapt"]
        settings = manifest["arguments"]
        expected = set(product(settings["profiles"], settings["parts"], settings["kinds"],
                               range(settings["seed"], settings["seed"] + settings["n"]), settings["methods"]))
        assert {identity(r) for r in rows} == expected
        by_key = {identity(r): r for r in rows}
        replayed = 0
        for row in rows:
            assert sum((row["in_place"], row["scrapped"], row["defective_finished"])) == 1
            assert row["frames"] == (0 if row["method"] == "ignore" else row["layers_printed"])
            assert row["adaptations"] <= 3
            if identity(row) in old:
                for field in ("in_place", "scrapped", "defective_finished", "layers_printed", "adaptations", "k0", "severity"):
                    assert row[field] == old[identity(row)][field], (identity(row), field)
                replayed += 1
            compact.append(dict(experiment=directory.name, **{k: v for k, v in row.items() if k != "actions"}))
        for split in ("clean", "failure"):
            for method in METHODS:
                selected = [r for r in rows if r["method"] == method and (r["kind"] == "clean") == (split == "clean")]
                if not selected:
                    continue
                ignored = [by_key[identity(r)[:-1] + ("ignore",)] for r in selected]
                baseline = [by_key[identity(r)[:-1] + ("ours",)] for r in selected]
                summary.append(dict(experiment=directory.name, split=split, method=method, n=len(selected),
                    in_place=sum(r["in_place"] for r in selected), scrapped=sum(r["scrapped"] for r in selected),
                    defective_finished=sum(r["defective_finished"] for r in selected),
                    harmful_if_ignored=sum(not b["in_place"] for b in ignored),
                    rescued_vs_ignore=sum(r["in_place"] and not b["in_place"] for r, b in zip(selected, ignored)),
                    lost_vs_ignore=sum(not r["in_place"] and b["in_place"] for r, b in zip(selected, ignored)),
                    gained_vs_ours=sum(r["in_place"] and not b["in_place"] for r, b in zip(selected, baseline)),
                    lost_vs_ours=sum(not r["in_place"] and b["in_place"] for r, b in zip(selected, baseline)),
                    pre_failure_alarms=sum(r["pre_failure_alarms"] for r in selected),
                    capped=sum(any(e["reason"] == "too many re-plans on one print" for e in r["actions"]) for r in selected)))
        comparisons = paired_comparison(rows)
        (args.out / (directory.name + "-paired.json")).write_text(json.dumps(comparisons, indent=2))
        (args.out / (directory.name + "-manifest.json")).write_text(json.dumps(manifest, indent=2))
        for comparison in comparisons:
            if comparison["treatment"] in ("functional", "guarded", "contour") and comparison["baseline"] in ("ignore", "ours", "predictive", "functional", "guarded"):
                for direction in ("gained", "lost"):
                    for case in comparison[direction]:
                        row = by_key[tuple(case) + (comparison["treatment"],)]
                        change_key = (directory.name, comparison["treatment"], *case)
                        if change_key not in changed:
                            changed[change_key] = dict(experiment=directory.name, method=comparison["treatment"], case=case,
                                                      comparisons=[], actions=row["actions"])
                        changed[change_key]["comparisons"].append(dict(baseline=comparison["baseline"], direction=direction))
        audit.append(dict(experiment=directory.name, directory=str(directory), runs=len(rows),
                          source_sha256=digest, baseline_replays_matched=replayed,
                          raw_sha256=hashlib.sha256((directory / "runs.jsonl").read_bytes()).hexdigest()))
    for name, records in (("outcomes", summary), ("runs", compact)):
        with (args.out / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2))
    (args.out / "changed-cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in changed.values()))
    for row in summary:
        print(f"{row['experiment']} {row['split']} {row['method']}: "
              f"{row['in_place']}/{row['n']} in place, {row['rescued_vs_ignore']} rescues, "
              f"{row['lost_vs_ignore']} lost vs ignore, {row['capped']} replan caps")


if __name__ == "__main__":
    main()
