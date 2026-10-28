"""Stress figures: where in the part the load is carried and how close it is to failing."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def stress_figure(case, title=""):
    """Failure index (1.0 = at the limit): top-down max over height, and a vertical section
    through the middle of the part."""
    g, fi = case.grid, case.fi3
    ext_xy = [g.x0, g.x0 + fi.shape[0] * g.h, g.y0, g.y0 + fi.shape[1] * g.h]
    top = np.nanmax(np.where(np.isnan(fi), -1, fi), axis=2)
    top = np.where(top < 0, np.nan, top)
    iy = fi.shape[1] // 2
    sec = fi[:, iy, :].T
    ext_xz = [g.x0, g.x0 + fi.shape[0] * g.h, g.z0, g.z0 + fi.shape[2] * g.hz]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#eeeeee")
    for ax, img, ext, lab in ((axs[0], top.T, ext_xy, "top view (max over height)"),
                              (axs[1], sec, ext_xz, "vertical section through the centre")):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=max(1.0, np.nanpercentile(fi, 99)))
        ax.set_title(lab)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)" if ext is ext_xy else "z (mm)")
    fig.colorbar(im, ax=axs, label="failure index (1 = at the limit)")
    fig.suptitle(f"{title} {case.name}: SF {min(case.safety_factor, 99):.1f}, "
                 f"deflection {case.max_disp_mm:.2f} mm" + (", LEAKS" if case.leak else ""))
    return fig
