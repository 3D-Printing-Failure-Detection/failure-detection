"""Matplotlib figures for the report and the UI."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geometry import features as F  # noqa: E402
from .geometry.ops import polys  # noqa: E402


def _draw(ax, region, **kw):
    for p in polys(region):
        for ring in [p.exterior, *p.interiors]:
            x, y = ring.xy
            ax.plot(x, y, **kw)


def layer_preview(layers, title="", n=4):
    """Outlines of a few layers (first, evenly spaced, last) as parsed from G-code, to check by
    eye that holes are holes and infill did not punch fake ones."""
    idx = sorted({int(round(i)) for i in np.linspace(0, len(layers) - 1, n)})
    fig, axs = plt.subplots(1, len(idx), figsize=(3.4 * len(idx), 3.6), squeeze=False)
    for ax, i in zip(axs[0], idx):
        lay = layers[i]
        _draw(ax, lay.region, color="#1f77b4", lw=1.0)
        ax.set_aspect("equal")
        ax.set_title(f"layer {i}  z={lay.z:.2f}  holes={F.base(lay)['n_main_holes']}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def event_figure(nominal, event, final_layers, title=""):
    """Left: expected vs measured contour at the alarm layer. Middle: nominal vs adapted
    cross-section at the top of the part. Right: interior area by height (nominal / final)."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    k = event.layer
    ax = axs[0]
    _draw(ax, nominal[k].region, color="#888", lw=1.2, label="expected")
    if event.measured is not None:
        _draw(ax, event.measured.region, color="#d62728", lw=1.6, label="measured")
    ax.set_aspect("equal"), ax.set_title(f"layer {k}: {event.label} ({event.deviation_mm:.1f} mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax = axs[1]
    top = len(nominal) - 1
    _draw(ax, nominal[top].region, color="#888", lw=1.2, label="nominal top")
    _draw(ax, final_layers[top].region, color="#2ca02c", lw=1.6, label="final top")
    ax.set_aspect("equal"), ax.set_title("top cross-section"), ax.legend(fontsize=8)
    ax = axs[2]
    z = [l.z for l in nominal]
    ax.plot(z, [F.base(l)["hole_area"] for l in nominal], color="#888", label="nominal")
    ax.plot(z, [F.base(l)["hole_area"] for l in final_layers], color="#2ca02c", label="final")
    ax.axvline(nominal[k].z, color="#d62728", ls=":", lw=1)
    ax.set_xlabel("height (mm)"), ax.set_ylabel("interior area (mm$^2$)")
    ax.set_title("interior area by height"), ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def metrics_bar(summary, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [s["method"] for s in summary]
    x = np.arange(len(names))
    ax.bar(x, [s["functional_in_place"] for s in summary], label="functional, no restart", color="#2ca02c")
    ax.bar(x, [s["restarted"] for s in summary], bottom=[s["functional_in_place"] for s in summary],
           label="scrapped + restarted", color="#ff7f0e")
    ax.bar(x, [s["defective"] for s in summary],
           bottom=[s["functional_in_place"] + s["restarted"] for s in summary],
           label="finished defective", color="#d62728")
    ax.set_xticks(x, names), ax.set_ylabel("share of scenarios"), ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
