"""P1S side-view renderer and dent/poke pair generator (recovery/p1s_view.py, recovery/p1s_dent_data.py)."""
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import pytest

pytest.importorskip("mujoco")

from recovery import p1s_dent_data as D
from recovery import p1s_view as V
from recovery.parts import get_part

TRAY = "parts/print/water-tray/water-tray.yaml"


def _render_tray(cam, top_layers=30, xy=(64.0, 64.0), drop=0.0):
    lay = get_part(TRAY).layers()[:top_layers]
    look = V.Look(filament_rgb=(1, 1, 1), plate_rgb1=(0, 0, 0), plate_rgb2=(0, 0, 0), chamber_rgb=(0, 0, 0),
                  ambient=0.6, headlight=0.4)
    with V.P1SScene(cam, look, {"part": V.stack_mesh(lay, bulge_mm=0.0)}, xy, 0.0, lay[-1].z, drop) as s:
        return s.render(["part"]), lay[-1].z


def test_lens_maps_invert_the_opencv_distortion():
    cam = V.P1SCamera.default()
    mx, my, rw, rh, f = V.remap_maps(cam)
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(0, 1279, 50), rng.uniform(0, 719, 50)]
    ray = np.c_[(mx[pts[:, 1].astype(int), pts[:, 0].astype(int)] - (rw / 2 - 0.5)) / f,
                (my[pts[:, 1].astype(int), pts[:, 0].astype(int)] - (rh / 2 - 0.5)) / f, np.ones(50)]
    back = cam.project((cam.R.T @ (ray.T * 100.0)).T + cam.centre)      # ray from the camera centre
    assert np.abs(back - pts.astype(int)).max() < 0.05


def test_rendered_part_sits_where_the_calibration_projects_it():
    cam = V.P1SCamera.default()
    (rgb, depth), top = _render_tray(cam)
    bright = rgb.mean(2) > 60
    plate_z = V.plate_z_mm(cam, top)
    corners = cam.project(D._corners_world((-15, -10, 15, 10), (64, 64), 0, plate_z, plate_z + top))
    ys, xs = np.nonzero(bright)
    assert abs(xs.min() - corners[:, 0].min()) < 3 and abs(xs.max() - corners[:, 0].max()) < 3
    assert abs(ys.min() - corners[:, 1].min()) < 3 and abs(ys.max() - corners[:, 1].max()) < 3


def test_moving_bed_keeps_the_top_at_the_nozzle_plane():
    """The P1S bed drops each layer: a taller print shows the same top edge and a lower plate."""
    cam = V.P1SCamera.default()
    (low, _), _ = _render_tray(cam, top_layers=15)
    (high, _), _ = _render_tray(cam, top_layers=40)
    top_row = [np.nonzero((im.mean(2) > 60).any(1))[0] for im in (low, high)]
    assert abs(top_row[0].min() - top_row[1].min()) <= 2            # same top
    assert top_row[1].max() > top_row[0].max() + 10                 # more wall below it
    assert V.plate_z_mm(cam, 8.0) == pytest.approx(cam.calib_z_mm - 8.0)


@pytest.mark.parametrize("kind", ["dent", "poke", "none"])
def test_pairs_are_labelled_and_the_mask_marks_the_change(kind):
    cfg = D.default_config(kinds=(kind,), weights=(1.0,), part_xy=(60.0, 70.0), yaw_sigma_deg=0.0, yaw_flip_p=0.0)
    for i in range(6):                         # a wall that faces the camera
        before, after, mask, lab, _ = D.make_pair(i, cfg)
        if kind == "none" or lab["outer_face_toward_camera"]:
            break
    assert before.shape == after.shape == (720, 1280, 3) and mask.shape == (720, 1280)
    assert lab["kind"] == kind and lab["pause_layer"] in range(27, 34)
    if kind == "none":
        assert lab["change_px"] == 0 and lab["wall"] is None and not lab["cavity_open_after"]
        assert np.array_equal(before, after)
        return
    assert lab["wall"] in D.WALLS and lab["visible"] and lab["change_px"] > 20
    x0, y0, x1, y1 = lab["change_bbox"]
    u, v = lab["contact_px"]
    assert x0 - 15 <= u <= x1 + 15 and y0 - 15 <= v <= y1 + 15        # change is where the tool touched
    assert lab["cavity_open_after"] == (kind == "poke")               # a poke leaks, a dent does not
    assert np.abs(after.astype(int) - before.astype(int))[mask > 0].mean() > 2


def test_generate_writes_images_labels_and_manifest(tmp_path):
    cfg = D.default_config(part_xy=(60.0, 70.0))
    m = D.generate(3, tmp_path, cfg, workers=1)
    assert m["pairs"] == 3 and not m["errors"]
    lines = (tmp_path / "labels.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for f in ("images/000000_before.jpg", "images/000000_after.jpg", "masks/000000.png", "manifest.json"):
        assert (tmp_path / f).exists()


def test_dent_model_crop_follows_the_part_and_decision_uses_the_change_area():
    torch = pytest.importorskip("torch")
    from recovery.models import dent as M
    cfg = D.default_config(kinds=("dent",), weights=(1.0,), part_xy=(60.0, 70.0), camera_jitter=False)
    before, after, mask, lab, _ = D.make_pair(0, cfg)
    poly = M.guide_polygon(V.P1SCamera.default(), lab["part_xy_mm"], lab["part_yaw_deg"], lab["top_z_mm"])
    x, box = M.make_input(before, after, poly)
    assert x.shape == (M.STORE, M.STORE, 7)
    assert M.crop(mask, box, M.STORE, M.cv2.INTER_LINEAR).max() > 0          # the change is inside the crop
    g = x[..., 6] > 0
    assert g.any() and not (g[0].any() or g[-1].any() or g[:, 0].any() or g[:, -1].any())   # whole guide in the crop
    area, pk = np.array([0, 3, 50]), np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    assert [M.KINDS[k] for k in M.decide(area, pk, 2)] == ["none", "poke", "dent"]
