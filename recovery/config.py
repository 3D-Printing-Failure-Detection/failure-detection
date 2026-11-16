"""Global constants shared by every module. All geometry is in millimetres,
with the part centred on (0, 0); G-code is offset by BED_CENTER."""
from dataclasses import dataclass

# Bambu Lab P1S: 256 x 256 mm bed, origin front-left, part usually arranged at the centre
BED_SIZE = (256.0, 256.0)
BED_CENTER = (128.0, 128.0)
# Chamber camera height above the plate at the first layer, measured by the team (Nov 2026). The
# bed drops as the print grows and the camera is fixed, so this is also its height above the top of
# every layer being printed. Used only to CHECK a calibration (and as an optional prior), never to map.
P1S_CAMERA_ABOVE_PRINT_PLANE_MM = 22.784
NOZZLE = 0.4
LINE_W = 0.42              # Bambu Studio default wall width for a 0.4 mm nozzle
FILAMENT_D = 1.75
FILAMENT_AREA = 3.14159265 * (FILAMENT_D / 2) ** 2


@dataclass(frozen=True)
class CameraCfg:
    px_per_mm: float = 4.0
    fov_mm: float = 128.0

    @property
    def size(self) -> int:
        return int(self.fov_mm * self.px_per_mm)


CAM = CameraCfg()
