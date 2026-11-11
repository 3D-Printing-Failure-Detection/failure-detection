"""Checkerboard-free calibration: recovers the camera from bed-edge clicks, and refuses bad input."""
import cv2
import numpy as np
import pytest

from recovery.calib_offline import (EDGES, fit_from_bed_edges, nominal_focal_px, refit_pose, save_points, load_points)
from tools.calibration_sensitivity import bed_error, f_from_diag_fov, make


def clicks(truth, n=8, noise=0.7, seed=0, bed=256.0):
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.08, 0.92, n) * bed
    edges = {"front": [(t, 0) for t in ts], "right": [(bed, t) for t in ts],
             "back": [(t, bed) for t in ts], "left": [(0, t) for t in ts]}
    out = {}
    for e, pts in edges.items():
        P = np.array([[x, y, 0.0] for x, y in pts])
        px, _ = cv2.projectPoints(P, truth.rvec, truth.tvec, truth.K, truth.dist)
        out[e] = (px.reshape(-1, 2) + rng.normal(0, noise, (n, 2))).tolist()
    return out


@pytest.mark.parametrize("true_fov", [110, 119])        # the datasheet prior is 110 either way
def test_fit_recovers_bed_geometry_despite_wrong_fov_prior(true_fov):
    truth = make(f_from_diag_fov(true_fov), k1=-0.25)
    cam, rep = fit_from_bed_edges(clicks(truth), diag_fov_deg=110)
    mean, worst = bed_error(truth, cam)
    assert mean < 1.0 and worst < 3.0                    # vs 17 mm mean with the datasheet FOV alone
    assert abs(cam.dist[0] - (-0.25)) < 0.15


def test_too_few_points_is_refused():
    truth = make(f_from_diag_fov(110))
    with pytest.raises(ValueError, match="at least"):
        fit_from_bed_edges(clicks(truth, n=3))


def test_bad_clicks_are_not_accepted():
    truth = make(f_from_diag_fov(110), k1=-0.2)
    pts = clicks(truth, noise=6.0)                         # sloppy clicking
    _, rep = fit_from_bed_edges(pts)
    assert not rep.ok


def test_pose_refit_reuses_lens_and_fixes_a_moved_camera():
    lens = make(f_from_diag_fov(110), k1=-0.25)
    moved = make(f_from_diag_fov(110), k1=-0.25, tilt_deg=47, shift=(6, 3, 2))
    P = np.array([[0, 0, 0], [256, 0, 0], [256, 256, 0], [0, 256, 0]], float)
    px, _ = cv2.projectPoints(P, moved.rvec, moved.tvec, moved.K, moved.dist)
    cam, rep = refit_pose(lens, px.reshape(4, 2))
    assert rep.ok and bed_error(moved, cam)[0] < 0.1


def test_points_roundtrip(tmp_path):
    pts = {e: [[1.0, 2.0], [3.0, 4.0]] for e in EDGES}
    save_points(pts, tmp_path / "p.json")
    assert load_points(tmp_path / "p.json") == pts


def test_nominal_focal_from_datasheet():
    assert 500 < nominal_focal_px(1280, 720, 110) < 530
