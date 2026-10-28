"""Load cases: what the part has to survive, simulated on the layer stack.

A case in the spec (parts/*.yaml -> `loads:`):

  - name: full of water, resting on a table
    fluid: {density_kg_m3: 1000, fill_height_mm: 36}   # liquid inside any enclosed cavity
    supports: [{face: bottom}]                          # fixed (default: the bottom face)
    loads:                                              # extra forces, spread over the selection
      - {box: [xmin, xmax, ymin, ymax, zmin, zmax], force_N: 30, direction: [0, 0, 1]}
    self_weight: true                                   # default true
    require: {safety_factor: 1.5, max_deflection_mm: 1.0, leak_free: true}

Selectors: {face: bottom|top} or {box: [...]} in part-centred mm (z from the bed), or both
(face AND box). Supports may add `dof: "xy"` to fix only some directions.

Physics included:
  * liquid test: flood-fill from outside; if liquid below the fill line can reach the outside
    through ANY gap, the part leaks. Otherwise hydrostatic pressure rho*g*(depth) acts on every
    wall/floor face that touches the liquid.
  * linear-elastic FE of the anisotropic printed material (weak interlayer plane).
  * pieces not connected to a support are reported as broken off.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import ndimage

from ..types import Layer
from .material import MATERIALS
from .model import FACE_NODES, Model, failure_index
from .voxel import VoxelGrid, stack_bounds, voxelize

G = 9810.0                      # mm / s^2


@dataclass
class CaseResult:
    name: str
    ok: bool
    violations: List[str]
    safety_factor: float = float("inf")      # from the 99th-percentile element (robust to corner spikes)
    safety_factor_peak: float = float("inf")
    max_disp_mm: float = 0.0
    leak: Optional[bool] = None
    capacity_mm3: Optional[float] = None
    floating_mm3: float = 0.0
    converged: bool = True
    iters: int = 0
    fi3: Optional[np.ndarray] = field(default=None, repr=False)     # (nx, ny, nz) failure index
    grid: Optional[VoxelGrid] = field(default=None, repr=False)


@dataclass
class StressReport:
    ok: bool
    cases: List[CaseResult]
    violations: List[str]

    def table(self) -> str:
        rows = [f"{'load case':44s} {'SF':>6s} {'SF peak':>8s} {'defl mm':>8s}  result"]
        for c in self.cases:
            rows.append(f"{c.name[:44]:44s} {min(c.safety_factor, 99):6.1f} {min(c.safety_factor_peak, 99):8.1f} "
                        f"{c.max_disp_mm:8.3f}  {'PASS' if c.ok else 'FAIL: ' + '; '.join(c.violations)}")
        return "\n".join(rows)


def liquid_cavity(solid, hz, z0, z_fill):
    """Voxels that hold liquid below z_fill: void regions NOT connected to the outside
    (through the sides or the bottom) below the fill line. Empty if the part leaks."""
    kf = int(np.clip(np.floor((z_fill - z0) / hz), 0, solid.shape[2]))
    cav = np.zeros_like(solid)
    if kf == 0:
        return cav
    sub = ~solid[:, :, :kf]
    pad = np.pad(sub, ((1, 1), (1, 1), (1, 0)), constant_values=True)     # outside air on sides + below
    lab, _ = ndimage.label(pad)
    outside = lab[0, 0, 0]
    inner = ((lab != outside) & (lab > 0))[1:-1, 1:-1, 1:]
    cav[:, :, :kf] = inner
    return cav


def select_nodes(model: Model, sel: dict):
    z = model.xyz[:, 2]
    m = np.ones(len(z), bool)
    if "face" in sel:
        ref = z.min() if sel["face"] == "bottom" else z.max()
        m &= np.isclose(z, ref)
    if "box" in sel:
        x0, x1, y0, y1, z0, z1 = sel["box"]
        eps = 1e-6
        m &= ((model.xyz[:, 0] >= x0 - eps) & (model.xyz[:, 0] <= x1 + eps)
              & (model.xyz[:, 1] >= y0 - eps) & (model.xyz[:, 1] <= y1 + eps)
              & (z >= z0 - eps) & (z <= z1 + eps))
    if "face" not in sel and "box" not in sel:
        m[:] = False
    return np.nonzero(m)[0]


def _fixed(model, supports):
    fixed = np.zeros(model.ndof, bool)
    sup_nodes = np.zeros(model.nn, bool)
    for s in supports:
        nodes = select_nodes(model, s)
        dofs = s.get("dof", "xyz")
        for c, ch in enumerate("xyz"):
            if ch in dofs:
                fixed[nodes * 3 + c] = True
        sup_nodes[nodes] = True
    return fixed, sup_nodes


def _fluid_forces(model, cavity, rho, z_fill, force):
    g, ei = model.grid, model.ei
    h, hz = g.h, g.hz
    shape = np.array(g.solid.shape)
    for axis in range(3):
        for s in (1, -1):
            nb = ei.copy()
            nb[:, axis] += s
            inside = ((nb >= 0) & (nb < shape)).all(1)
            wet = np.zeros(model.ne, bool)
            wet[inside] = cavity[tuple(nb[inside].T)]
            idx = np.nonzero(wet)[0]
            if not len(idx):
                continue
            zf = g.z0 + (ei[idx, 2] + (1 if (axis == 2 and s == 1) else 0) + (0 if axis == 2 else 0.5)) * hz
            p = rho * G * np.maximum(z_fill - zf, 0.0)                     # MPa
            area = h * hz if axis < 2 else h * h
            f = p * area / 4.0
            for ln in FACE_NODES[(axis, s)]:
                np.add.at(force, model.enode[idx, ln] * 3 + axis, -s * f)   # liquid pushes into the solid


def _run_case(layers: List[Layer], case: dict, spec: dict, mat, layer_h, h) -> CaseResult:
    name = case.get("name", "load case")
    m = max(1, int(round(h / layer_h)))
    grid = voxelize(layers, h, m, "mid")
    viol, leak, cap, rho_f, z_fill = [], None, None, 0.0, 0.0
    fluid = case.get("fluid")
    if fluid:
        zmax = layers[-1].z
        z_fill = float(fluid.get("fill_height_mm", fluid.get("fill_frac", 0.9) * zmax))
        rho_f = float(fluid.get("density_kg_m3", 1000.0)) * 1e-12            # kg/m^3 -> t/mm^3
        fine = voxelize(layers, 0.5, 1, "min", bounds=stack_bounds(layers))
        cap = float(liquid_cavity(fine.solid, fine.hz, fine.z0, z_fill).sum() * fine.h ** 2 * fine.hz)
        cap_lo = ((spec.get("container") or {}).get("capacity_mm3") or [0])[0]
        leak = cap < max(0.5 * cap_lo, 200.0)
        if leak:
            viol.append("leaks: liquid below the fill line escapes through the part")

    supports = case.get("supports") or [{"face": "bottom"}]
    model = Model(grid, mat)
    fixed, sup_nodes = _fixed(model, supports)
    if not sup_nodes.any():
        return CaseResult(name, False, viol + ["support selection touches no material"], leak=leak,
                          capacity_mm3=cap)
    from scipy import ndimage as ndi
    labels, _ = ndi.label(grid.solid, structure=np.ones((3, 3, 3), int))
    elab = labels[tuple(model.ei.T)]
    keep = np.unique(elab[sup_nodes[model.enode].any(1)])
    floating = ~np.isin(elab, keep)
    floating_mm3 = float(floating.sum() * h * h * grid.hz)
    if floating.any():
        solid2 = grid.solid.copy()
        solid2[tuple(model.ei[floating].T)] = False
        grid = VoxelGrid(solid2, grid.h, grid.hz, grid.x0, grid.y0, grid.z0)
        model = Model(grid, mat)
        fixed, sup_nodes = _fixed(model, supports)
        viol.append(f"{floating_mm3:.0f} mm3 of the part is not connected to a support (breaks off)")

    force = np.zeros(model.ndof)
    if fluid and not leak:
        cav = liquid_cavity(grid.solid, grid.hz, grid.z0, z_fill)
        _fluid_forces(model, cav, rho_f, z_fill, force)
    for ld in case.get("loads", []):
        nodes = select_nodes(model, ld)
        if not len(nodes):
            viol.append("a load region selects no material")
            continue
        d = np.array(ld.get("direction", [0, 0, -1]), float)
        d /= np.linalg.norm(d)
        for c in range(3):
            np.add.at(force, nodes * 3 + c, ld["force_N"] * d[c] / len(nodes))
    if case.get("self_weight", True):
        w = mat.rho * G * grid.h * grid.h * grid.hz                         # N per element
        np.add.at(force, (model.enode * 3 + 2).ravel(), -w / 8.0)

    u, conv, iters = model.solve(force, fixed)
    fi = failure_index(model.stress(u), mat)
    sf = 1.0 / max(float(np.percentile(fi, 99)), 1e-9)
    sf_peak = 1.0 / max(float(fi.max()), 1e-9)
    disp = float(np.linalg.norm(u.reshape(-1, 3), axis=1).max())

    req = case.get("require", {})
    if not conv:
        viol.append("solver did not converge (result unreliable)")
    if sf < req.get("safety_factor", 1.5):
        viol.append(f"safety factor {sf:.2f} < {req.get('safety_factor', 1.5)}")
    if "max_deflection_mm" in req and disp > req["max_deflection_mm"]:
        viol.append(f"deflection {disp:.2f} mm > {req['max_deflection_mm']}")
    if leak and not req.get("leak_free", bool(fluid)):
        viol = [v for v in viol if not v.startswith("leaks")]

    fi3 = np.full(grid.solid.shape, np.nan, np.float32)
    fi3[tuple(model.ei.T)] = fi
    return CaseResult(name, not viol, viol, sf, sf_peak, disp, leak, cap, floating_mm3, conv, iters,
                      fi3, grid)


def stress_test(layers: List[Layer], spec: dict, layer_h: float, h: float = 1.5,
                material: Optional[str] = None) -> StressReport:
    """Run every load case in spec['loads'] on this layer stack (nominal, printed, or planned)."""
    cases = spec.get("loads") or []
    mat = MATERIALS[material or spec.get("material", "PLA")]
    results = [_run_case(layers, c, spec, mat, layer_h, h) for c in cases]
    viol = [f"[{r.name}] {v}" for r in results for v in r.violations]
    return StressReport(not viol, results, viol)
