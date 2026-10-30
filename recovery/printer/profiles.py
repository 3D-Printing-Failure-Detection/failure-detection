"""Illustrative rigs, not manufacturer-calibrated digital twins. Units: mm."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrinterProfile:
    name: str
    kinematics: str
    bed_mm: tuple[float, float]
    height_mm: float
    line_width_mm: float
    moving_mass_kg: float
    camera_position_m: tuple[float, float, float]
    filament_rgb: tuple[float, float, float]
    noise_std: float


PROFILES = {
    "cartesian": PrinterProfile("cartesian", "cartesian", (220, 220), 250, .42, .25,
                                (0, -.12, .36), (.85, .82, .75), 1.0),
    "corexy": PrinterProfile("corexy", "corexy", (256, 256), 256, .45, .16,
                             (.12, -.15, .40), (.75, .83, .87), 1.5),
    "bedslinger": PrinterProfile("bedslinger", "bedslinger", (180, 180), 180, .48, .45,
                                 (-.10, -.18, .34), (.87, .72, .59), 2.0),
}
