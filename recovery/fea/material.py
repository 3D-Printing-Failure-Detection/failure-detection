"""FDM material models. Printed parts are anisotropic: filaments inside a layer are strong,
the bond BETWEEN layers is the weak plane. Units are N, mm, MPa, tonne (so density is t/mm^3
and gravity is 9810 mm/s^2).

The numbers are conservative design values in the range reported for FDM PLA/PETG/ABS in the
literature and datasheets. They are NOT measured on your printer: print a few tensile coupons
(flat and upright) on the P1S and replace them for quantitative work.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    E_xy: float        # MPa, in-plane modulus (along the layer)
    E_z: float         # MPa, modulus across layers
    nu_xy: float
    nu_xz: float
    S_xy: float        # MPa, in-plane strength (compared with von Mises stress)
    S_zt: float        # MPa, interlayer tensile strength (weak plane)
    S_zs: float        # MPa, interlayer shear strength
    rho: float         # tonne / mm^3

    def stiffness(self) -> np.ndarray:
        """6x6 transversely isotropic stiffness, Voigt order (xx, yy, zz, yz, xz, xy),
        engineering shear strains. z is the build direction."""
        E1, E3 = self.E_xy, self.E_z
        G12 = E1 / (2 * (1 + self.nu_xy))
        G13 = E1 / (2 * (1 + self.nu_xz))
        S = np.zeros((6, 6))
        S[0, 0] = S[1, 1] = 1 / E1
        S[2, 2] = 1 / E3
        S[0, 1] = S[1, 0] = -self.nu_xy / E1
        S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -self.nu_xz / E1
        S[3, 3] = S[4, 4] = 1 / G13
        S[5, 5] = 1 / G12
        return np.linalg.inv(S)


MATERIALS = {
    "PLA":  Material("PLA",  3000.0, 2400.0, 0.35, 0.30, 45.0, 20.0, 18.0, 1.24e-9),
    "PETG": Material("PETG", 2100.0, 1800.0, 0.38, 0.33, 40.0, 25.0, 20.0, 1.27e-9),
    "ABS":  Material("ABS",  2000.0, 1700.0, 0.35, 0.30, 35.0, 20.0, 17.0, 1.04e-9),
}
