"""How wrong is a camera model built from datasheet numbers instead of a checkerboard?

Illustrative, not a measurement: a 'true' camera is projected through OpenCV, then points are
back-projected onto the bed with a camera that is wrong by a plausible amount. Prints the mm error
over the 256 x 256 mm bed. The pipeline's alarm threshold is 1 mm.
Run: PYTHONPATH=. python tools/calibration_sensitivity.py
"""
import cv2
import numpy as np

from recovery.camera import BedCamera

W, H = 1280, 720
DIAG = np.hypot(W, H)


def f_from_diag_fov(deg):
    return DIAG / 2 / np.tan(np.radians(deg) / 2)


def make(f, k1=0.0, tilt_deg=45.0, dist_mm=330.0, yaw=0.0, shift=(0, 0, 0)):
    """Camera looking at the bed centre (128,128,0) from the front, tilt_deg below horizontal."""
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    t = np.radians(tilt_deg)
    centre = np.array([128.0, 128.0, 0.0])
    cam_pos = centre + dist_mm * np.array([0, -np.cos(t), np.sin(t)]) + np.array(shift, float)
    z = (centre - cam_pos) / np.linalg.norm(centre - cam_pos)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z]) @ cv2.Rodrigues(np.array([0, 0, np.radians(yaw)]))[0]
    return BedCamera(K, np.array([k1, 0, 0, 0, 0.0]), cv2.Rodrigues(R)[0], -R @ cam_pos)


def bed_error(true, assumed):
    g = np.linspace(28, 228, 9)
    pts = np.array([[x, y, 0.0] for x in g for y in g])
    px, _ = cv2.projectPoints(pts, true.rvec, true.tvec, true.K, true.dist)
    und = cv2.undistortPoints(px, assumed.K, assumed.dist, P=assumed.K).reshape(-1, 2)
    Hinv = np.linalg.inv(assumed.plane_to_image(0.0))
    q = (Hinv @ np.c_[und, np.ones(len(und))].T).T
    err = np.linalg.norm(q[:, :2] / q[:, 2:] - pts[:, :2], axis=1)
    return err.mean(), err.max()


def main():
    truth = make(f_from_diag_fov(110), k1=-0.25)
    cases = {
        "perfect": truth,
        "datasheet FOV right (110 deg), distortion ignored": make(f_from_diag_fov(110)),
        "FOV 119 deg (reseller figure) instead of 110": make(f_from_diag_fov(119), k1=-0.25),
        "focal length off by 3 %": make(f_from_diag_fov(110) * 1.03, k1=-0.25),
        "pose: tilt off by 1 deg": make(f_from_diag_fov(110), k1=-0.25, tilt_deg=46),
        "pose: camera position off by 5 mm": make(f_from_diag_fov(110), k1=-0.25, shift=(5, 0, 0)),
        "pose: yaw off by 1 deg": make(f_from_diag_fov(110), k1=-0.25, yaw=1),
    }
    print(f"{'assumed camera vs truth':55s} mean mm   max mm")
    for name, cam in cases.items():
        m, M = bed_error(truth, cam)
        print(f"{name:55s} {m:7.2f} {M:8.2f}")


if __name__ == '__main__':
    main()
