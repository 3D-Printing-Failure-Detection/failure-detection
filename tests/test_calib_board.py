"""Board calibration on synthetic P1S frames: a 10 mm checkerboard covering the plate, seen by a camera
placed like the real one (22.8 mm above the print plane, looking ~9 deg down), rendered with lens distortion."""
import cv2
import numpy as np
import pytest

from recovery import config as C
from recovery.calib_board import calibrate_from_board, detect_board
from recovery.camera import BedCamera
from tests.test_moving_bed import p1s_camera

W, H = 1280, 720
SQ = 10.0
THICK = 2.0                                   # board surface above the plate (foam board)


def render(cam_K, cam_dist, R_bc, t_bc, n=26, sq=SQ, ss=2):
    """Frame of an n x n checkerboard whose plane is mapped to the camera by (R_bc, t_bc) (board mm -> camera mm)."""
    ys, xs = np.mgrid[0:H * ss, 0:W * ss].astype(np.float32)
    pix = (np.c_[xs.ravel(), ys.ravel()] + 0.5) / ss - 0.5
    und = cv2.undistortPoints(pix.reshape(-1, 1, 2), cam_K, cam_dist).reshape(-1, 2)
    Hb = np.column_stack([R_bc[:, 0], R_bc[:, 1], t_bc])           # board (u, v, 1) -> normalized ray
    q = (np.linalg.inv(Hb) @ np.c_[und, np.ones(len(und))].T).T
    uv = q[:, :2] / q[:, 2:]
    s = 1.0 / q[:, 2]                                               # depth along the ray (> 0: in front)
    inside = (uv >= 0).all(1) & (uv <= n * sq).all(1) & (s > 0)
    dark = ((np.floor(uv[:, 0] / sq) + np.floor(uv[:, 1] / sq)) % 2 == 0)
    img = np.full(len(uv), 40.0)
    img[inside] = np.where(dark[inside], 25.0, 215.0)
    img = cv2.resize(img.reshape(H * ss, W * ss).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    img += np.random.default_rng(0).normal(0, 2.0, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def flat_frame(cam: BedCamera, origin=(3.0, 3.0)):
    """Board lying flat on the plate (its corner (0,0) at bed `origin`), surface THICK above the plate."""
    R, _ = cv2.Rodrigues(cam.rvec)
    t = R @ np.array([origin[0], origin[1], THICK]) + cam.tvec.ravel()
    return render(cam.K, cam.dist, R, t)


def tilted_frame(cam: BedCamera, seed):
    """Board held up in front of the camera at a random angle."""
    rng = np.random.default_rng(seed)
    rv = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)])
    R, _ = cv2.Rodrigues(np.array([np.pi, 0, 0]) + rv)              # board facing the camera
    centre = np.array([rng.uniform(-60, 60), rng.uniform(-30, 30), rng.uniform(260, 380)])
    t = centre - R @ np.array([130.0, 130.0, 0])
    return render(cam.K, cam.dist, R, t)


def truth(z, f=845.0, k1=-0.30):
    cam = p1s_camera(z, f=f, k1=k1)
    cam.dist = np.array([k1, 0.10, 0, 0, 0.0])
    return cam


def markers_for(cam, pts_mm, origin=(3.0, 3.0)):
    P = np.array([[x, y, THICK] for x, y in pts_mm], float)
    px, _ = cv2.projectPoints(P, cam.rvec, cam.tvec, cam.K, cam.dist)
    return [dict(px=p.tolist(), mm=list(m)) for p, m in zip(px.reshape(-1, 2), pts_mm)]


MARKS = [(103.0, 103.0), (153.0, 133.0), (123.0, 163.0)]     # board corners (origin + multiples of 10 mm)


def print_plane_error(true_cam, cam):
    g = np.linspace(90, 170, 5)
    P = np.array([[x, y, true_cam.calib_z_mm] for x in g for y in g])
    px, _ = cv2.projectPoints(P, true_cam.rvec, true_cam.tvec, true_cam.K, true_cam.dist)
    und = cv2.undistortPoints(px, cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    q = (np.linalg.inv(cam.plane_to_image(cam.layer_plane_z(4.0))) @ np.c_[und, np.ones(len(und))].T).T
    return float(np.linalg.norm(q[:, :2] / q[:, 2:] - P[:, :2], axis=1).max())


@pytest.fixture(scope="module")
def frames():
    t10, t30 = truth(10.0), truth(30.0)
    return dict(t10=t10, f10=flat_frame(t10), f30=flat_frame(t30),
                tilted=[tilted_frame(t10, s) for s in range(6)])


def test_detector_indexes_a_grazing_board_consistently(frames):
    t10 = frames["t10"]
    ij, px = detect_board(frames["f10"])
    assert len(ij) >= 100
    # every detected corner is a true board corner, and its index maps to it by one lattice transform
    g = np.array([[x, y] for x in range(27) for y in range(27)], float)
    P = np.c_[g * SQ + 3.0, np.full(len(g), THICK)]
    tp, _ = cv2.projectPoints(P, t10.rvec, t10.tvec, t10.K, t10.dist)
    tp = tp.reshape(-1, 2)
    nearest = np.array([np.argmin(np.linalg.norm(tp - p, axis=1)) for p in px])
    assert np.median(np.linalg.norm(tp[nearest] - px, axis=1)) < 0.5
    true_ij = g[nearest].astype(int)
    # majority lattice map: the true step for each detected axis, then the common offset
    det = {tuple(k): t for k, t in zip(ij, true_ij)}
    steps = []
    for e in ((1, 0), (0, 1)):
        diffs = [tuple(det[(k[0] + e[0], k[1] + e[1])] - t) for k, t in det.items() if (k[0] + e[0], k[1] + e[1]) in det]
        steps.append(max(set(diffs), key=diffs.count))
    M = np.array(steps).T
    assert abs(round(np.linalg.det(M))) == 1 and np.abs(M).sum() == 2        # both axes are single square edges
    off = [tuple(t - M @ k) for k, t in zip(ij, true_ij)]
    best = max(set(off), key=off.count)
    assert off.count(best) / len(off) > 0.9


def test_flat_only_fallback_recovers_the_camera_and_checks_z(frames):
    t10 = frames["t10"]
    cam, rep = calibrate_from_board([], [(frames["f10"], 10.0), (frames["f30"], 30.0)], markers_for(t10, MARKS),
                                    board_thickness_mm=THICK)
    assert rep.mode.startswith("flat-only") and rep.ok, rep.text()
    assert abs(rep.heights[1][1] - rep.heights[0][1] - 20.0) < 1.0
    assert abs(cam.pose()["height_above_print_plane_mm"] - C.P1S_CAMERA_ABOVE_PRINT_PLANE_MM) < 1.0
    assert cam.calib_z_mm == 10.0
    assert print_plane_error(t10, cam) < 1.5


def test_a_wrong_bed_z_is_caught(frames):
    t10 = frames["t10"]
    _, rep = calibrate_from_board([], [(frames["f10"], 10.0), (frames["f30"], 20.0)], markers_for(t10, MARKS),
                                  board_thickness_mm=THICK)
    assert not rep.ok and any("bed moved" in n for n in rep.notes)


def test_tilted_frames_are_preferred_and_fit_the_lens(frames):
    t10 = frames["t10"]
    cam, rep = calibrate_from_board(frames["tilted"], [(frames["f10"], 10.0), (frames["f30"], 30.0)],
                                    markers_for(t10, MARKS), board_thickness_mm=THICK)
    assert rep.mode == "tilted-frame" and rep.n_tilted >= 3, rep.text()
    assert abs(cam.K[0, 0] - 845.0) / 845.0 < 0.03
    assert rep.ok, rep.text()
    assert print_plane_error(t10, cam) < 1.5


def test_one_flat_frame_without_tilted_frames_is_refused(frames):
    with pytest.raises(ValueError, match="2 different bed Z"):
        calibrate_from_board([], [(frames["f10"], 10.0)], markers_for(frames["t10"], MARKS))


def test_markers_off_the_grid_are_refused(frames):
    bad = markers_for(frames["t10"], MARKS)
    bad[1]["mm"] = [108.0, 73.0]                          # half a square off
    with pytest.raises(ValueError, match="markers"):
        calibrate_from_board([], [(frames["f10"], 10.0), (frames["f30"], 30.0)], bad, board_thickness_mm=THICK)


def test_cli_fits_from_a_shots_folder(frames, tmp_path):
    import argparse
    import json
    from recovery.calib_board import main
    for name, img in (("flat_00.png", frames["f10"]), ("flat_01.png", frames["f30"])):
        cv2.imwrite(str(tmp_path / name), img)
    (tmp_path / "shots.json").write_text(json.dumps(dict(tilted=[], flat=[dict(file="flat_00.png", z=10),
                                                                           dict(file="flat_01.png", z=30)])))
    (tmp_path / "markers.json").write_text(json.dumps(dict(markers=markers_for(frames["t10"], MARKS))))
    a = argparse.Namespace(ip=None, code=None, dir=str(tmp_path), markers=None, square_mm=SQ, board_thickness=THICK,
                           camera_height=C.P1S_CAMERA_ABOVE_PRINT_PLANE_MM, out=str(tmp_path / "calib.npz"), force=False)
    main(a)
    cam = BedCamera.load(tmp_path / "calib.npz")
    assert cam.calib_z_mm == 10.0 and (tmp_path / "calib_overlay.png").exists()


def test_the_same_corner_clicked_twice_is_refused(frames):
    m = markers_for(frames["t10"], MARKS[:1]) * 2
    with pytest.raises(ValueError, match="DIFFERENT"):
        calibrate_from_board([], [(frames["f10"], 10.0), (frames["f30"], 30.0)], m, board_thickness_mm=THICK)


def test_frames_of_different_sizes_are_refused(frames):
    small = cv2.resize(frames["f30"], (960, 540))
    with pytest.raises(ValueError, match="same size"):
        calibrate_from_board([], [(frames["f10"], 10.0), (small, 30.0)], markers_for(frames["t10"], MARKS))
