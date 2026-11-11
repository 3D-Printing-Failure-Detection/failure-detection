"""CalculiX backend: hand the SAME voxel mesh, material, supports and loads to the open-source
CalculiX solver (`ccx`, GPL, a widely used finite-element code) instead of the built-in CG.

The built-in solver stays as an offline fallback; `solver="auto"` uses CalculiX when `ccx` is on
PATH (or $CCX_PATH). Nothing here is certification: CalculiX is open source and heavily
cross-checked against benchmarks, but no standards body has certified it. Use `fea.verify` to
reproduce analytical benchmarks on your install, and tensile coupons for the material numbers.

Install (any one):  apt install calculix-ccx   |   conda install -c conda-forge calculix
                    brew install calculix-ccx  (tap names vary; or build from calculix.de)

Mesh: each solid voxel becomes one 8-node brick (C3D8, full integration). Material: orthotropic
engineering constants (z = build axis). Loads: the nodal force vector computed by cases.py
(fluid pressure, face loads, self weight) goes in as *CLOAD; supports as *BOUNDARY.
Output: nodal displacements and the 8 integration-point stresses per brick, averaged to one
element-centre stress in the order used by the rest of the package (xx, yy, zz, yz, xz, xy).
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .material import Material


def find_ccx() -> Optional[str]:
    env = os.environ.get("CCX_PATH")
    if env and shutil.which(env):
        return shutil.which(env)
    for name in ("ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19", "ccx_static", "ccx_MT"):
        p = shutil.which(name)
        if p:
            return p
    return None


def available() -> bool:
    return find_ccx() is not None


def write_inp(path: Path, model, force: np.ndarray, fixed: np.ndarray, mat: Material):
    """Write a CalculiX input deck for a linear static analysis of `model` (see model.Model)."""
    E1, E3 = mat.E_xy, mat.E_z
    G12 = E1 / (2 * (1 + mat.nu_xy))
    G13 = E1 / (2 * (1 + mat.nu_xz))
    lines = ["*HEADING", "recovery voxel model, units N-mm-MPa", "*NODE"]
    lines += [f"{i + 1},{x:.6f},{y:.6f},{z:.6f}" for i, (x, y, z) in enumerate(model.xyz)]
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    lines += [f"{e + 1}," + ",".join(str(int(n) + 1) for n in model.enode[e]) for e in range(model.ne)]
    lines += ["*MATERIAL, NAME=FDM", "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
              f"{E1:.6g},{E1:.6g},{E3:.6g},{mat.nu_xy:.6g},{mat.nu_xz:.6g},{mat.nu_xz:.6g},"
              f"{G12:.6g},{G13:.6g},", f"{G13:.6g}",
              "*SOLID SECTION, ELSET=EALL, MATERIAL=FDM", "*STEP", "*STATIC"]
    fx = np.nonzero(fixed)[0]
    if len(fx):
        lines.append("*BOUNDARY")
        lines += [f"{d // 3 + 1},{d % 3 + 1},{d % 3 + 1},0" for d in fx]
    nz = np.nonzero((force != 0) & ~fixed)[0]
    if len(nz):
        lines.append("*CLOAD")
        lines += [f"{d // 3 + 1},{d % 3 + 1},{force[d]:.9g}" for d in nz]
    lines += ["*NODE PRINT, NSET=NALL, FREQUENCY=1", "U",
              "*EL PRINT, ELSET=EALL, FREQUENCY=1", "S", "*END STEP"]
    # NSET=NALL must exist: define it after the nodes, before the step
    deck = "\n".join(lines)
    deck = deck.replace("*ELEMENT, TYPE=C3D8, ELSET=EALL",
                        f"*NSET, NSET=NALL, GENERATE\n1,{model.nn},1\n*ELEMENT, TYPE=C3D8, ELSET=EALL", 1)
    path.write_text(deck + "\n")


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_dat(path: Path, nn: int, ne: int) -> Tuple[np.ndarray, np.ndarray]:
    """-> (u (3*nn,), element stress (ne, 6) in xx,yy,zz,yz,xz,xy)."""
    u = np.zeros(3 * nn)
    sig = np.zeros((ne, 6))
    cnt = np.zeros(ne)
    mode = None
    for ln in path.read_text().splitlines():
        low = ln.lower()
        if "displacements" in low:
            mode = "u"
            continue
        if "stresses" in low:
            mode = "s"
            continue
        parts = ln.split()
        if not parts or not re.fullmatch(r"\d+", parts[0]):
            continue
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            continue
        if mode == "u" and len(vals) == 4:
            i = int(vals[0]) - 1
            u[3 * i:3 * i + 3] = vals[1:]
        elif mode == "s" and len(vals) == 8:                       # elem, ip, sxx syy szz sxy sxz syz
            e = int(vals[0]) - 1
            sxx, syy, szz, sxy, sxz, syz = vals[2:]
            sig[e] += (sxx, syy, szz, syz, sxz, sxy)
            cnt[e] += 1
    if (cnt == 0).any():
        raise RuntimeError("CalculiX output is missing element stresses")
    return u, sig / cnt[:, None]


def solve(model, force, fixed, mat: Material, workdir: Optional[str] = None,
          timeout=1800) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Run ccx. -> (u, element_stress, converged)."""
    ccx = find_ccx()
    if ccx is None:
        raise RuntimeError("CalculiX (ccx) not found; install it or use solver='builtin'")
    ctx = tempfile.TemporaryDirectory() if workdir is None else None
    d = Path(workdir or ctx.name)
    d.mkdir(parents=True, exist_ok=True)
    write_inp(d / "job.inp", model, force, fixed, mat)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 1)))
    run = subprocess.run([ccx, "job"], cwd=d, capture_output=True, text=True, timeout=timeout, env=env)
    dat = d / "job.dat"
    if run.returncode != 0 or not dat.exists():
        raise RuntimeError("ccx failed:\n" + (run.stdout + run.stderr)[-1500:])
    u, sig = parse_dat(dat, model.nn, model.ne)
    if ctx:
        ctx.cleanup()
    return u, sig, True
