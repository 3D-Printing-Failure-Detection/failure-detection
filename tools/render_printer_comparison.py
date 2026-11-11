"""Export actual MuJoCo RGB views and the calibrated controller input.

This is an illustrative scene at a specified layer, not an evaluation rollout.
"""
import argparse
import os
import sys
if sys.platform.startswith("linux"):          # headless servers; macOS/Windows use their native GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recovery.parts import get_part
from recovery.printer.mujoco import MujocoPrinter
from recovery.printer.profiles import PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/validation/printer-views.png")
    args = parser.parse_args()
    nominal = get_part("mug").layers()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, profile in enumerate(PROFILES):
        with MujocoPrinter(nominal, profile=profile) as printer:
            printer.printed = list(nominal[:80])
            top = printer.camera_image()
            axes[0, column].imshow(printer.last_raw_image[..., ::-1])
            axes[1, column].imshow(top[..., ::-1])
            axes[0, column].set_title(profile)
            axes[1, column].set_title("Calibrated layer view (128 mm)")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("MuJoCo inspection rigs — illustrative mug scene at z = 20 mm")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
