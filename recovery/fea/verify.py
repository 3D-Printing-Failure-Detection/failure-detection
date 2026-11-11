"""Check a solver against closed-form results on THIS machine, and check mesh convergence.

  python -m recovery.cli fea-verify [--solver calculix]

1. Cantilever beam (Euler-Bernoulli + shear correction): tip deflection of a rectangular bar
   fixed at one end under an end load. Voxel hex8 elements are stiff in bending, so the error is
   reported, not hidden; it should shrink as the mesh is refined.
2. Uniform tension: stress = F / A exactly (patch test), must match to a fraction of a percent.
3. `convergence()` re-runs a real part's load cases at several voxel sizes so you can see
   whether the safety factor has stopped moving.
"""
import numpy as np
import shapely.geometry as sg

from ..types import Layer
from .cases import resolve_solver, stress_test
from .material import MATERIALS


def _bar(L, w, t, layer_h):
    """Solid bar: x in [0, L] horizontal is a rod lying along y? Build direction z is the LONG axis
    here (a vertical bar), loaded sideways (bending) or along z (tension)."""
    n = int(round(L / layer_h))
    box = sg.box(-w / 2, -t / 2, w / 2, t / 2)
    return [Layer((i + 1) * layer_h, box) for i in range(n)]


def tension_patch_test(solver=None, h=1.0, layer_h=0.2, F=200.0, w=12.0, t=12.0, L=20.0):
    layers = _bar(L, w, t, layer_h)
    spec = dict(material="PLA", loads=[dict(name="tension", self_weight=False,
                                            supports=[dict(face="bottom")],
                                            loads=[dict(face="top", force_N=F, direction=[0, 0, 1])],
                                            require=dict(safety_factor=0.01))])
    rep = stress_test(layers, spec, layer_h, h, solver=solver)
    c = rep.cases[0]
    mat = MATERIALS["PLA"]
    exact_disp = F * L / (w * t * mat.E_z)
    return dict(name="uniform tension", solver=rep.solver, computed=c.max_disp_mm, exact=exact_disp,
                error_pct=100 * (c.max_disp_mm - exact_disp) / exact_disp)


def cantilever_test(solver=None, h=1.0, layer_h=0.2, F=5.0, w=12.0, t=12.0, L=40.0):
    """Vertical bar, fixed at the bed, sideways end load in x. Bending uses E in the layer plane
    (the bar axis is z, so bending stress runs along z: E_z governs)."""
    layers = _bar(L, w, t, layer_h)
    spec = dict(material="PLA", loads=[dict(name="cantilever", self_weight=False,
                                            supports=[dict(face="bottom")],
                                            loads=[dict(face="top", force_N=F, direction=[1, 0, 0])],
                                            require=dict(safety_factor=0.01))])
    rep = stress_test(layers, spec, layer_h, h, solver=solver)
    c = rep.cases[0]
    mat = MATERIALS["PLA"]
    I = t * w ** 3 / 12                       # bending about y for load along x
    G13 = mat.E_xy / (2 * (1 + mat.nu_xz))
    kappa = 5 / 6
    exact = F * L ** 3 / (3 * mat.E_z * I) + F * L / (kappa * G13 * w * t)
    return dict(name="cantilever tip", solver=rep.solver, computed=c.max_disp_mm, exact=exact,
                error_pct=100 * (c.max_disp_mm - exact) / exact)


def verify(solver=None, hs=(2.0, 1.0)):
    rows = []
    for h in hs:
        for fn in (tension_patch_test, cantilever_test):
            r = fn(solver, h=h)
            r["h_mm"] = h
            rows.append(r)
    return rows


def format_rows(rows):
    out = [f"{'benchmark':18s} {'solver':9s} {'h mm':>5s} {'computed':>10s} {'exact':>10s} {'error %':>8s}"]
    for r in rows:
        out.append(f"{r['name']:18s} {r['solver']:9s} {r['h_mm']:5.1f} {r['computed']:10.5f} "
                   f"{r['exact']:10.5f} {r['error_pct']:8.2f}")
    return "\n".join(out)


def convergence(layers, spec, layer_h, hs=(2.0, 1.5, 1.0, 0.75), solver=None):
    """Safety factor of each load case versus voxel size. -> list of (h, [SF per case])."""
    rows = []
    for h in hs:
        rep = stress_test(layers, spec, layer_h, h, solver=solver)
        rows.append((h, [c.safety_factor for c in rep.cases]))
    return rows


def converged(rows, tol=0.10):
    """True if the finest two meshes agree on every case's safety factor within `tol` (relative)."""
    if len(rows) < 2:
        return False
    a, b = np.array(rows[-2][1]), np.array(rows[-1][1])
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(b, 1e-9)))
