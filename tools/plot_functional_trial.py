"""Plot the audited outcomes; scraps never count as successful recoveries."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="audit output directory containing outcomes.csv")
    args = parser.parse_args()
    with (args.directory / "outcomes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    methods = ["ignore", "ours", "measured", "predictive", "functional"]
    labels = ["Ignore", "Current policy", "1: measured", "1+2: predictive", "1+2+3: margin"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, name, title in zip(axes, ["function-main-v1", "function-heldout-v1"],
                               ["Three rigs, seed 10", "Cartesian, fresh seed 20"]):
        selected = {r["method"]: r for r in rows if r["experiment"] == name and r["split"] == "failure"}
        y, left = np.arange(len(methods)), np.zeros(len(methods))
        for field, label, color in [("in_place", "Functional completion", "#2b8c7b"),
                                    ("scrapped", "Scrapped", "#b7bdc5"),
                                    ("defective_finished", "Defective completion", "#db8044")]:
            values = np.array([int(selected[m][field]) / int(selected[m]["n"]) for m in methods])
            ax.barh(y, values, left=left, label=label, color=color, height=.65)
            for i, (value, m) in enumerate(zip(values, methods)):
                if value >= .06:
                    ax.text(left[i] + value / 2, i, selected[m][field], ha="center", va="center", fontsize=9)
            left += values
        ax.set_title(title + f" ({selected['ours']['n']} failure runs per policy)", fontsize=11)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of failure runs (numbers are counts)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(args.directory / "outcomes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
