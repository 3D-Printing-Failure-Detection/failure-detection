"""Linear-elastic hexahedral (voxel) finite elements, matrix-free preconditioned CG.

Every solid voxel is an 8-node brick with the anisotropic material of material.py. Because
all bricks are the same size, one 24x24 element matrix serves them all, and K @ u is done by
gather -> multiply -> scatter, so a 100k-element part needs no global matrix in memory.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .material import Material
from .voxel import VoxelGrid

# node order of a brick: (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)
OFF = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
SGN = OFF * 2 - 1
FACE_NODES = {(0, 1): [1, 2, 5, 6], (0, -1): [0, 3, 4, 7],
              (1, 1): [2, 3, 6, 7], (1, -1): [0, 1, 4, 5],
              (2, 1): [4, 5, 6, 7], (2, -1): [0, 1, 2, 3]}


def _dN(xi, eta, zeta, hx, hy, hz):
    d = np.zeros((8, 3))
    for a, (sx, sy, sz) in enumerate(SGN):
        d[a, 0] = sx * (1 + sy * eta) * (1 + sz * zeta) / 8 * 2 / hx
        d[a, 1] = sy * (1 + sx * xi) * (1 + sz * zeta) / 8 * 2 / hy
        d[a, 2] = sz * (1 + sx * xi) * (1 + sy * eta) / 8 * 2 / hz
    return d


def _B(d):
    B = np.zeros((6, 24))
    for a in range(8):
        c = 3 * a
        B[0, c] = d[a, 0]
        B[1, c + 1] = d[a, 1]
        B[2, c + 2] = d[a, 2]
        B[3, c + 1], B[3, c + 2] = d[a, 2], d[a, 1]      # gamma_yz
        B[4, c], B[4, c + 2] = d[a, 2], d[a, 0]          # gamma_xz
        B[5, c], B[5, c + 1] = d[a, 1], d[a, 0]          # gamma_xy
    return B


def hex8(hx, hy, hz, C):
    g = 1 / np.sqrt(3)
    detJ = hx * hy * hz / 8
    Ke = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _B(_dN(xi, eta, zeta, hx, hy, hz))
                Ke += B.T @ C @ B * detJ
    return Ke, _B(_dN(0.0, 0.0, 0.0, hx, hy, hz))


class Model:
    def __init__(self, grid: VoxelGrid, mat: Material):
        self.grid, self.mat = grid, mat
        self.C = mat.stiffness()
        h, hz = grid.h, grid.hz
        self.Ke, self.B0 = hex8(h, h, hz, self.C)
        ei = np.argwhere(grid.solid)
        if len(ei) == 0:
            raise ValueError("empty model")
        self.ei, self.ne = ei, len(ei)
        nx, ny, nz = grid.solid.shape
        gid = (((ei[:, 0:1] + OFF[:, 0]) * (ny + 1) + (ei[:, 1:2] + OFF[:, 1])) * (nz + 1)
               + (ei[:, 2:3] + OFF[:, 2]))
        uniq, inv = np.unique(gid.ravel(), return_inverse=True)
        self.enode = inv.reshape(self.ne, 8)
        self.nn, self.ndof = len(uniq), 3 * len(uniq)
        gi = uniq // ((ny + 1) * (nz + 1))
        gj = (uniq // (nz + 1)) % (ny + 1)
        gk = uniq % (nz + 1)
        self.xyz = np.stack([grid.x0 + gi * h, grid.y0 + gj * h, grid.z0 + gk * hz], 1)
        self.edof = (self.enode[:, :, None] * 3 + np.arange(3)).reshape(self.ne, 24)

    def solve(self, force, fixed, rtol=1e-6, maxiter=6000):
        """K u = f with `fixed` dofs held at zero. Returns (u, converged, iterations)."""
        free = ~fixed
        n, nf = self.ndof, int(free.sum())
        ed, flat, Ke = self.edof, self.edof.ravel(), self.Ke

        def K(u):
            return np.bincount(flat, weights=(u[ed] @ Ke).ravel(), minlength=n)

        diag = np.bincount(flat, weights=np.tile(np.diag(Ke), self.ne), minlength=n)[free]
        full = np.zeros(n)

        def A(x):
            full[free] = x
            return K(full)[free]

        count = [0]

        def cb(_):
            count[0] += 1

        op = LinearOperator((nf, nf), matvec=A, dtype=float)
        pre = LinearOperator((nf, nf), matvec=lambda r: r / diag, dtype=float)
        try:
            x, info = cg(op, force[free], M=pre, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
        except TypeError:                                   # SciPy < 1.12
            x, info = cg(op, force[free], M=pre, tol=rtol, maxiter=maxiter, callback=cb)
        u = np.zeros(n)
        u[free] = x
        return u, info == 0, count[0]

    def stress(self, u):
        """Element-centre stress (Ne, 6): xx, yy, zz, yz, xz, xy in MPa."""
        eps = u[self.edof] @ self.B0.T
        return eps @ self.C.T


def failure_index(sig, mat: Material):
    """>= 1 means the element is at its limit. Two modes: von Mises against in-plane strength,
    and the interlayer (weak plane) interaction of tension across layers and shear."""
    sx, sy, sz, tyz, txz, txy = sig.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                 + 3 * (tyz ** 2 + txz ** 2 + txy ** 2))
    fi_plane = vm / mat.S_xy
    fi_inter = np.sqrt((np.maximum(sz, 0) / mat.S_zt) ** 2 + (txz ** 2 + tyz ** 2) / mat.S_zs ** 2)
    return np.maximum(fi_plane, fi_inter)
