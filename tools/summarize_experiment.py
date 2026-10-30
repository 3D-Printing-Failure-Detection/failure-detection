"""Audit complete paired results and export small, portable report artifacts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from recovery.eval.mujoco_eval import source_digest, summarize, write_csv


def audited_rows(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    args = manifest["arguments"]
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    expected = set(itertools.product(args["profiles"], args["parts"], args["kinds"],
                                    range(args["seed"], args["seed"] + args["n"]), args["methods"]))
    key = lambda r: (r["profile"], r["part"], r["kind"], r["seed"], r["method"])
    if manifest["status"] != "complete" or manifest["errors"]:
        raise ValueError(f"{directory}: experiment did not complete")
    if len(rows) != len(expected) or {key(row) for row in rows} != expected:
        raise ValueError(f"{directory}: missing or duplicate experimental conditions")
    if not manifest["source_sha256"] == manifest["final_source_sha256"] == source_digest():
        raise ValueError(f"{directory}: source changed during or after the run")
    for row in rows:
        if row["clear_failure_on_adapt"] != args["clear_failure_on_adapt"]:
            raise ValueError("failure semantics differ from the manifest")
        if row["method"] != "ignore" and row["frames"] != row["layers_printed"]:
            raise ValueError("a monitored layer was skipped")
        if row["in_place"] and (row["outcome"] != "finished" or row["layers_printed"] != row["n_layers"]):
            raise ValueError("success was assigned to an incomplete print")
    failure_params = {}
    for row in rows:
        scenario = (row["part"], row["kind"], row["seed"])
        parameters = (row["k0"], row["severity"])
        if failure_params.setdefault(scenario, parameters) != parameters:
            raise ValueError("unpaired failure parameters across rigs or policies")
    return rows, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--transient", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/validation"))
    args = parser.parse_args()
    persistent, main_manifest = audited_rows(args.persistent)
    transient, ablation_manifest = audited_rows(args.transient)
    if main_manifest["arguments"]["clear_failure_on_adapt"] or not ablation_manifest["arguments"]["clear_failure_on_adapt"]:
        raise ValueError("persistent/transient experiments are reversed")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "persistent-summary.csv", summarize(persistent))
    write_csv(args.out / "transient-summary.csv", summarize(transient))
    ignore = {(r["profile"], r["part"], r["kind"], r["seed"]): r for r in persistent if r["method"] == "ignore"}
    comparisons = []
    for profile in main_manifest["arguments"]["profiles"]:
        for assumption, rows in [("persistent", persistent), ("cleared_on_adapt", transient)]:
            selected = [r for r in rows if r["profile"] == profile and r["method"] == "ours" and r["kind"] != "clean"]
            harmful = [r for r in selected if not ignore[(profile, r["part"], r["kind"], r["seed"])]["in_place"]]
            comparisons.append(dict(profile=profile, assumption=assumption, n=len(selected),
                in_place=sum(r["in_place"] for r in selected),
                scrapped=sum(r["scrapped"] for r in selected),
                defective_finished=sum(r["defective_finished"] for r in selected),
                harmful=len(harmful), recovered_harmful=sum(r["in_place"] for r in harmful),
                adapted=sum(r["adaptations"] > 0 for r in selected)))
    write_csv(args.out / "assumption-comparison.csv", comparisons)
    by_kind = []
    for profile, kind, method in itertools.product(main_manifest["arguments"]["profiles"],
                                                  main_manifest["arguments"]["kinds"],
                                                  main_manifest["arguments"]["methods"]):
        group = [r for r in persistent if (r["profile"], r["kind"], r["method"]) == (profile, kind, method)]
        by_kind.append(dict(profile=profile, kind=kind, method=method, n=len(group),
                            in_place=sum(r["in_place"] for r in group),
                            scrapped=sum(r["scrapped"] for r in group),
                            defective_finished=sum(r["defective_finished"] for r in group)))
    write_csv(args.out / "outcomes-by-kind.csv", by_kind)
    receipt = dict(verified_source_sha256=source_digest(), main_runs=len(persistent),
        ablation_runs=len(transient),
        checks=["exact experimental condition coverage", "unique rows", "complete manifests",
                "unchanged source digest", "all monitored layers imaged", "no success assigned to incomplete prints",
                "paired failure parameters"],
        raw_files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for directory in (args.persistent, args.transient)
                   for path in (directory / "runs.jsonl", directory / "manifest.json")},
        profiles=main_manifest["profiles"], packages=main_manifest["packages"],
        main_arguments=main_manifest["arguments"], ablation_arguments=ablation_manifest["arguments"],
        comparisons=comparisons)
    (args.out / "experiment-audit.json").write_text(json.dumps(receipt, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    profiles = main_manifest["arguments"]["profiles"]
    x = np.arange(len(profiles))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ignored_rates = []
    for profile in profiles:
        group = [r for r in persistent if r["profile"] == profile and r["method"] == "ignore" and r["kind"] != "clean"]
        ignored_rates.append(100 * sum(r["in_place"] for r in group) / len(group))
    ax.bar(x-.25, ignored_rates, .25, label="Ignore failure", color="#777777")
    for offset, assumption, label, color in [(0, "persistent", "Recovery: persistent cause", "#176a93"),
                                            (.25, "cleared_on_adapt", "Recovery: cause cleared", "#b07a1f")]:
        values = [next(r["in_place"] / r["n"] * 100 for r in comparisons
                       if r["profile"] == profile and r["assumption"] == assumption) for profile in profiles]
        ax.bar(x+offset, values, .25, label=label, color=color)
    ax.set(xticks=x, xticklabels=profiles, ylim=(0, 100), ylabel="Functional completion in place (%)",
           title="Paired synthetic failures: same perception and policy across rigs")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "recovery-comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
