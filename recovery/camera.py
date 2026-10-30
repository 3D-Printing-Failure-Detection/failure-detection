"""Top-down camera model: render layer stacks to images, rasterise expected
layers, and convert pixel masks back to mm contours. Also real-camera calibration."""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon

from . import config as C
from .geometry.ops import clean, polys
from .types import Layer

CAM = C.CAM


def mm_to_px(pts: np.ndarray, cam=CAM) -> np.ndarray:
    pts = np.asarray(pts, float)
    return np.stack([pts[:, 0] * cam.px_per_mm + cam.size / 2,
                     cam.size / 2 - pts[:, 1] * cam.px_per_mm], 1)


def px_to_mm(pts: np.ndarray, cam=CAM) -> np.ndarray:
    pts = np.asarray(pts, float)
    return np.stack([(pts[:, 0] - cam.size / 2) / cam.px_per_mm,
                     (cam.size / 2 - pts[:, 1]) / cam.px_per_mm], 1)


def rasterise(region, cam=CAM, value=255, out=None) -> np.ndarray:
    m = np.zeros((cam.size, cam.size), np.uint8) if out is None else out
    for p in polys(region):
        component = np.zeros_like(m)
        ext = np.round(mm_to_px(np.array(p.exterior.coords), cam) * 16).astype(np.int32)
        cv2.fillPoly(component, [ext], value, lineType=cv2.LINE_AA, shift=4)
        for r in p.interiors:
            # Subpixel voids between buffered extrusion segments otherwise each
            # erase a full pixel, turning a solid hatch into a nearly empty image.
            if Polygon(r).area < (0.5 / cam.px_per_mm) ** 2:
                continue
            h = np.round(mm_to_px(np.array(r.coords), cam) * 16).astype(np.int32)
            cv2.fillPoly(component, [h], 0, lineType=cv2.LINE_AA, shift=4)
        np.maximum(m, component, out=m)
    return m


def expected_mask(layer: Layer, cam=CAM) -> np.ndarray:
    return (rasterise(layer.region, cam) > 127).astype(np.uint8)


def render(recent: List[Layer], rng: np.random.Generator, cam=CAM) -> np.ndarray:
    """Synthetic camera image (BGR uint8). `recent` = [base layer, ..., current],
    older layers are drawn dimmer so the current layer is the brightest."""
    n = cam.size
    bed = rng.uniform(0.10, 0.28)
    img = np.full((n, n), bed, np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    img += rng.uniform(-0.05, 0.05) * (xx - 0.5) + rng.uniform(-0.05, 0.05) * (yy - 0.5)
    fil = rng.uniform(0.78, 0.95)
    # base layer dimmest, then progressively brighter, current layer brightest
    older = [0.50 * fil] + [fil * f for f in np.linspace(0.62, 0.72, max(len(recent) - 2, 0))]
    shades = older[: len(recent) - 1] + [fil]
    for lay, s in zip(recent, shades):
        m = rasterise(lay.region, cam).astype(np.float32) / 255.0
        img = img * (1 - m) + s * m
    # texture, shading, noise, blur
    tex = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 2.0) * 0.04
    img += tex * (img > bed + 0.1)
    img *= 1.0 - 0.15 * (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 2)          # vignette
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 0.9))
    img += rng.normal(0, rng.uniform(0.01, 0.035), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    tint = rng.uniform(0.85, 1.0, 3).astype(np.float32)
    return (np.clip(img[..., None] * tint, 0, 1) * 255).astype(np.uint8)


def mask_to_region(mask: np.ndarray, cam=CAM, min_area=4.0, simplify=0.12):
    """Binary mask -> shapely region in mm (outer contours with holes)."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return Polygon()
    hier = hier[0]
    out = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or len(c) < 4:
            continue
        shell = px_to_mm(c[:, 0, :].astype(float), cam)
        holes = []
        j = hier[i][2]
        while j != -1:
            if len(cnts[j]) >= 4:
                holes.append(px_to_mm(cnts[j][:, 0, :].astype(float), cam))
            j = hier[j][0]
        pg = Polygon(shell, holes)
        if pg.area >= min_area:
            out.append(pg)
    if not out:
        return Polygon()
    from shapely.ops import unary_union
    # contours run through pixel centres: grow by half a pixel to get the true outline
    g = clean(unary_union(out).buffer(0.5 / cam.px_per_mm, quad_segs=2))
    return g.simplify(simplify) if simplify else g


# ---------- real camera: pose-based top-down view ----------
@dataclass
class BedCamera:
    """Calibrated real camera (e.g. the P1S chamber camera, which looks at the bed at an angle).
    K, dist: intrinsics. rvec, tvec: pose of the BED frame (mm, origin front-left, z up) in the
    camera frame. top_down() resamples a frame into the same top-down view the simulator
    renders, for the plane at the current print height."""
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

    def save(self, path):
        np.savez(path, K=self.K, dist=self.dist, rvec=self.rvec, tvec=self.tvec)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["K"], d["dist"], d["rvec"], d["tvec"])

    def plane_to_image(self, z=0.0) -> np.ndarray:
        """3x3 homography: bed point (X, Y, 1) on the plane at height z -> image pixel."""
        R, _ = cv2.Rodrigues(self.rvec)
        t = R[:, 2] * z + self.tvec.reshape(3)
        return self.K @ np.column_stack([R[:, 0], R[:, 1], t])

    def top_down(self, frame: np.ndarray, z: float, center_mm, cam=CAM) -> np.ndarray:
        """Frame -> top-down image (cam.size px, cam.px_per_mm) centred on `center_mm`
        (bed coords of the part centre), sampled on the plane at height z (the layer top)."""
        und = cv2.undistort(frame, self.K, self.dist)
        ppm, n = cam.px_per_mm, cam.size
        cx, cy = center_mm
        T = np.array([[1 / ppm, 0, cx - n / (2 * ppm)],
                      [0, -1 / ppm, cy + n / (2 * ppm)],
                      [0, 0, 1]], float)                       # target px -> bed mm
        M = self.plane_to_image(z) @ T                         # target px -> source px
        return cv2.warpPerspective(und, M, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def _board_points(pattern, square_mm, origin_mm):
    return np.array([[origin_mm[0] + i * square_mm, origin_mm[1] + j * square_mm, 0.0]
                     for j in range(pattern[1]) for i in range(pattern[0])], np.float32)


def _find_corners(image, pattern):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))


def calibrate_bed_camera(frames, pattern=(9, 6), square_mm=10.0, board_origin_mm=(100.0, 100.0),
                         pose_frame=None):
    """Build a BedCamera from checkerboard photos.

    frames: several photos of the board in different places/tilts (intrinsics + distortion);
    pose_frame: one photo with the board lying FLAT on the bed with its first inner corner at
    `board_origin_mm` (bed coordinates, mm) -- this fixes the bed frame. Defaults to frames[-1].
    Use a sheet of paper printed at 100% scale; measure the square size with calipers.
    """
    pose_frame = frames[-1] if pose_frame is None else pose_frame
    obj0 = _board_points(pattern, square_mm, (0.0, 0.0))
    objs, imgs = [], []
    for f in frames:
        c = _find_corners(f, pattern)
        if c is not None:
            objs.append(obj0), imgs.append(c)
    if len(imgs) < 5:
        raise RuntimeError(f"need >= 5 frames with a visible board, found {len(imgs)}")
    h, w = frames[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (w, h), None, None)
    c = _find_corners(pose_frame, pattern)
    if c is None:
        raise RuntimeError("board not found in the pose frame")
    ok, rvec, tvec = cv2.solvePnP(_board_points(pattern, square_mm, board_origin_mm), c, K, dist)
    if not ok:
        raise RuntimeError("pose estimation failed")
    print(f"intrinsics RMS reprojection error: {rms:.2f} px (aim for < 0.7)")
    return BedCamera(K, dist, rvec, tvec)
