"""Checkerboard-free calibration of the P1S chamber camera.

Why it can work: the bed is a known 256 x 256 mm rectangle and the camera is fixed in the
chamber. Clicking points along the four bed edges in ONE frame gives enough constraints to fit
the focal length, one radial-distortion term and the pose, with priors from the datasheet
(110 deg diagonal FOV, 1280 x 720) keeping the fit well-posed. The fit REPORTS its own error
in mm and refuses to produce a calibration it cannot vouch for.

Entry points, cheapest first:
  refit_pose(profile, corners_px)   known lens (from a profile) + 4 corner clicks -> new pose
  fit_from_bed_edges(points, ...)   no profile: fit lens + pose from edge clicks

Pixel points are plain JSON {"front": [[u,v],..], "right": .., "back": .., "left": ..}
produced by the click page that `calibrate-bed` writes (pick_points_html).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import BedCamera

EDGES = ("front", "right", "back", "left")     # bed frame: origin front-left, +X right, +Y back
NOMINAL = dict(width=1280, height=720, diag_fov_deg=110.0)
MAX_RESIDUAL_MM = 1.0                          # same as the alarm threshold
F_STARTS = (0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.1)   # x datasheet focal length: the cost has local minima
HEIGHT_MATCH_MM = 3.0                          # a start whose camera height is this close to the measured one wins


def nominal_focal_px(width, height, diag_fov_deg):
    return float(np.hypot(width, height) / 2 / np.tan(np.radians(diag_fov_deg) / 2))


def _K(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _corners_from_edges(points):
    """Least-squares image lines per edge, intersected -> 4 corners (front-left, front-right, ...)."""
    lines = {}
    for e in EDGES:
        p = np.asarray(points[e], float)
        vx, vy, x0, y0 = cv2.fitLine(p.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines[e] = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])

    def meet(a, b):
        c = np.cross(lines[a], lines[b])
        return c[:2] / c[2]
    return np.array([meet("front", "left"), meet("front", "right"), meet("back", "right"), meet("back", "left")])


def _bed_corners_mm(bed):
    return np.array([[0, 0, 0], [bed[0], 0, 0], [bed[0], bed[1], 0], [0, bed[1], 0]], np.float32)


def _to_bed(cam: BedCamera, px):
    und = cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    try:
        H = np.linalg.inv(cam.plane_to_image(0.0))
    except np.linalg.LinAlgError:      # degenerate pose during the search
        return np.full((len(und), 2), np.nan)
    q = (H @ np.c_[und, np.ones(len(und))].T).T
    return q[:, :2] / q[:, 2:]


def _edge_residuals(cam, points, bed):
    """Signed mm distance of every clicked point from the bed edge it was clicked on."""
    out = []
    for e in EDGES:
        xy = _to_bed(cam, points[e])
        out.append({"front": xy[:, 1], "back": xy[:, 1] - bed[1], "left": xy[:, 0], "right": xy[:, 0] - bed[0]}[e])
    return np.concatenate(out)


@dataclass
class FitReport:
    rms_mm: float
    max_mm: float
    focal_px: float
    k1: float
    n_points: int
    ok: bool
    notes: list

    def text(self):
        s = (f"edge residual RMS {self.rms_mm:.2f} mm, max {self.max_mm:.2f} mm over {self.n_points} points; "
             f"f={self.focal_px:.0f}px k1={self.k1:+.3f}  ->  {'OK' if self.ok else 'NOT ACCEPTABLE'}")
        return "\n".join([s] + [f"  - {n}" for n in self.notes])


def _check_points(points, min_per_edge):
    missing = [e for e in EDGES if e not in points or len(points[e]) < min_per_edge]
    if missing:
        raise ValueError(f"need at least {min_per_edge} points on each bed edge; short: {missing}")


def _camera_height(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return float((-R.T @ np.asarray(tvec, float).reshape(3))[2])


def _camera_above_bed(cam):
    return _camera_height(cam.rvec, cam.tvec) > 0


def _marker_residuals(cam, markers):
    if not markers:
        return np.zeros(0)
    got = _to_bed(cam, [m["px"] for m in markers])
    return (got - np.array([m["mm"] for m in markers], float)).ravel()


def edge_straightness(points):
    """Per-edge RMS (px) of the raw clicks about their best straight line. A lens bows edges, but one
    edge far worse than the rest usually means a mis-click or a feature that is not the plate edge."""
    out = {}
    for e in EDGES:
        p = np.asarray(points[e], np.float32)
        vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        out[e] = float(np.sqrt(np.mean(((p[:, 0] - x0) * vy - (p[:, 1] - y0) * vx) ** 2)))
    return out


def _fit_once(points, markers, image_size, bed, f_start, f0, fit_distortion, fit_k2, height=None):
    w, h = image_size
    corners = _corners_from_edges(points)
    ok, rvec, tvec = cv2.solvePnP(_bed_corners_mm(bed), corners.astype(np.float32), _K(f_start, w, h), None)
    if not ok:
        return None
    x0 = np.r_[np.log(f_start / f0), 0.0, 0.0, rvec.ravel(), tvec.ravel()]

    def build(x):
        return BedCamera(_K(f0 * np.exp(x[0]), w, h),
                         np.array([x[1] if fit_distortion else 0.0, x[2] if fit_k2 else 0.0, 0, 0, 0.0]),
                         x[3:6].reshape(3, 1), x[6:9].reshape(3, 1))

    def resid(x):
        cam = build(x)
        r = np.r_[_edge_residuals(cam, points, bed), _marker_residuals(cam, markers)]
        r = np.clip(np.nan_to_num(r, nan=500.0, posinf=500.0, neginf=-500.0), -500, 500)
        # weak priors keep f and distortion plausible when the clicks alone cannot separate them
        prior = [] if height is None else [(_camera_height(x[3:6], x[6:9]) - height[0]) / height[1]]
        return np.r_[r, x[0] / 0.6, x[1] / 1.0, (x[2] / 1.5 if fit_k2 else 0.0), prior]

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=1.0, max_nfev=300)
    cam = build(sol.x)
    return sol.cost, cam


def fit_from_bed_edges(points, image_size=(1280, 720), bed=(256.0, 256.0), diag_fov_deg=None,
                       fit_distortion=True, min_per_edge=6, markers=None, fit_k2=True,
                       calib_z_mm=None, camera_height_mm=None, height_sigma_mm=None):
    """Fit f, k1 (and k2) and pose from clicked bed-edge points, plus optional `markers`
    [{"px": [u, v], "mm": [X, Y]}, ...] at known bed positions (very effective: they constrain the
    interior of the view). Several focal-length starts are tried because the cost has local minima.
    calib_z_mm: bed Z during the frame (stored in the camera; see BedCamera.calib_z_mm).
    camera_height_mm: measured camera height above the print plane; it is always checked against the
    fit, and with height_sigma_mm it also constrains the fit (camera at camera_height + calib_z
    above the plate), which helps most when the plate is seen at a grazing angle.
    Returns (BedCamera, FitReport)."""
    _check_points(points, min_per_edge)
    markers = markers or []
    w, h = image_size
    fov = diag_fov_deg or NOMINAL["diag_fov_deg"]
    f0 = nominal_focal_px(w, h, fov)
    height = None
    if height_sigma_mm and camera_height_mm is not None:
        if calib_z_mm is None:
            raise ValueError("a camera-height prior needs calib_z_mm (bed Z during the frame)")
        height = (camera_height_mm + calib_z_mm, height_sigma_mm)
    runs = [r for r in (_fit_once(points, markers, image_size, bed, f0 * m, f0, fit_distortion, fit_k2, height)
                        for m in F_STARTS) if r]
    if not runs:
        raise RuntimeError("could not get an initial pose from the bed corners")
    above = [r for r in runs if _camera_above_bed(r[1])]
    pool = above or runs
    if camera_height_mm is not None and calib_z_mm is not None:
        # on a grazing view several poses explain the clicks almost equally well (e.g. a camera twice as
        # high with a shorter lens); the independently measured height picks the physical one
        want = camera_height_mm + calib_z_mm
        pool = [r for r in pool if abs(_camera_height(r[1].rvec, r[1].tvec) - want) <= HEIGHT_MATCH_MM] or pool
    cam = min(pool, key=lambda r: r[0])[1]
    cam.calib_z_mm = calib_z_mm
    r = np.r_[_edge_residuals(cam, points, bed), _marker_residuals(cam, markers)]
    r = np.clip(np.nan_to_num(r, nan=500.0, posinf=500.0, neginf=-500.0), -500, 500)
    notes = []
    f, k1 = float(cam.K[0, 0]), float(cam.dist[0])
    rms, worst = float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())
    st = edge_straightness(points)
    notes.append("clicks' straightness (px RMS about a line): " + ", ".join(f"{e} {v:.1f}" for e, v in st.items()))
    per = []
    for e in EDGES:
        xy = _to_bed(cam, points[e])
        d = {"front": xy[:, 1], "back": xy[:, 1] - bed[1], "left": xy[:, 0], "right": xy[:, 0] - bed[0]}[e]
        per.append(f"{e} {np.sqrt(np.nanmean(np.clip(d, -500, 500) ** 2)):.1f}")
    notes.append("residual mm RMS per edge: " + ", ".join(per) + "  (edges seen at a grazing angle are the noisiest: "
                 "one pixel there is several mm)")
    typical = np.median(list(st.values()))
    for e, v in st.items():
        if v > max(4.0, 3 * typical):
            notes.append(f"the '{e}' edge is much more curved than the others: re-click it (is it really the plate edge?)")
    if not above:
        notes.append("every solution puts the camera BELOW the bed: left/right (or front/back) are probably mirrored")
    if rms > MAX_RESIDUAL_MM:
        swapped = {**points, "left": points["right"], "right": points["left"]}
        if not markers:
            alt = [x for x in (_fit_once(swapped, markers, image_size, bed, f0 * m, f0, fit_distortion, fit_k2)
                               for m in F_STARTS) if x]
            if alt:
                ra = _edge_residuals(min(alt, key=lambda x: x[0])[1], swapped, bed)
                if np.sqrt(np.mean(np.clip(ra, -500, 500) ** 2)) < 0.5 * rms:
                    notes.append("swapping the LEFT and RIGHT labels fits much better: they look mirrored "
                                 "(left/right are as seen standing at the front of the printer, facing it)")
    if calib_z_mm is not None:
        notes.append(cam.pose_text())
    height_warning = cam.check_height(camera_height_mm)
    if height_warning:
        notes.append(height_warning)
    ok_fit = bool(above and worst <= MAX_RESIDUAL_MM * 2 and rms <= MAX_RESIDUAL_MM and not height_warning)
    if ok_fit:
        notes.append("residuals only prove the clicks are consistent; also check the overlay image and run a clean `watch`")
    return cam, FitReport(rms, worst, f, k1, len(r), ok_fit, notes)


def refit_pose(profile: BedCamera, corners_px, bed=(256.0, 256.0), calib_z_mm=None):
    """Keep a known lens (K, dist); recompute only the pose from the 4 bed corners
    (front-left, front-right, back-right, back-left). Use after the camera or printer was moved.
    calib_z_mm: bed Z in this new frame (defaults to the profile's)."""
    corners = np.asarray(corners_px, np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(_bed_corners_mm(bed), corners, profile.K, profile.dist)
    if not ok:
        raise RuntimeError("pose estimation failed")
    cam = BedCamera(profile.K, profile.dist, rvec, tvec,
                    profile.calib_z_mm if calib_z_mm is None else calib_z_mm)
    back = _to_bed(cam, corners)
    err = np.linalg.norm(back - _bed_corners_mm(bed)[:, :2], axis=1)
    notes = [] if err.max() <= MAX_RESIDUAL_MM else [f"corners reproject {err.max():.1f} mm off: lens profile does not fit this unit"]
    return cam, FitReport(float(np.sqrt(np.mean(err ** 2))), float(err.max()), float(cam.K[0, 0]),
                          float(cam.dist[0]), 4, err.max() <= MAX_RESIDUAL_MM, notes)


def verify(cam: BedCamera, points, bed=(256.0, 256.0)):
    r = _edge_residuals(cam, points, bed)
    return FitReport(float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max()), float(cam.K[0, 0]), float(cam.dist[0]),
                     len(r), bool(np.abs(r).max() <= MAX_RESIDUAL_MM * 2), [])


def draw_overlay(cam: BedCamera, frame, bed=(256.0, 256.0), step=32):
    """Frame with the predicted bed outline and a grid: it must sit on the real plate edges."""
    out = frame.copy()
    lines = [[(x, 0), (x, bed[1])] for x in np.arange(0, bed[0] + 1, step)] + \
            [[(0, y), (bed[0], y)] for y in np.arange(0, bed[1] + 1, step)]
    for a, b in lines:
        seg = np.linspace(a, b, 40)
        pts = np.c_[seg, np.zeros(40)].astype(np.float64)
        px, _ = cv2.projectPoints(pts, cam.rvec, cam.tvec, cam.K, cam.dist)
        edge = a[0] in (0, bed[0]) and b[0] in (0, bed[0]) or a[1] in (0, bed[1]) and b[1] in (0, bed[1])
        px = np.nan_to_num(px.reshape(-1, 2), nan=-1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
        cv2.polylines(out, [px.astype(np.int32)], False, (0, 255, 0) if edge else (0, 160, 255), 2 if edge else 1)
    return out


def save_points(points, path, markers=None):
    d = {e: np.asarray(points[e], float).round(1).tolist() for e in EDGES}
    if markers:
        d["markers"] = markers
    Path(path).write_text(json.dumps(d, indent=1))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SystemExit(f"no {path}: save it from the click page first")
    except (UnicodeDecodeError, ValueError):
        raise SystemExit(f"{path} is not valid JSON (an editor such as TextEdit adds formatting and curly quotes): "
                         f"use the click page's Save button, or in Terminal: pbpaste > {path}")


def load_points(path):
    d = _read_json(path)
    missing = [e for e in EDGES if e not in d]
    if missing:
        raise SystemExit(f"{path} has no clicks for: {', '.join(missing)} (use the calibrate-bed click page)")
    return {e: d[e] for e in EDGES}


def load_markers(path):
    return _read_json(path).get("markers", [])


def pick_points_html(image_name, out_path, intro=None, markers_only=False):
    """Write a dependency-free page: click >=4 points along each bed edge (or markers), then save the JSON.
    markers_only: a page for board corners at known bed X,Y only (calibrate-board), saving markers.json."""
    save_name = "markers.json" if markers_only else "points.json"
    intro = intro or ("Choose an edge, then click at least <b>4 points along that plate edge</b>, spread from end to "
                      "end (front = edge nearest the door/you, left/right as you face the printer, back = far edge). "
                      "Put one click on each corner too. Then <b>Save points.json</b>.")
    title = "Click the board corners" if markers_only else "Click the bed edges"
    options = ("<option value=marker>marker (known X,Y mm)" if markers_only else
               "<option>front<option>right<option>back<option>left<option value=marker>marker (known X,Y mm)")
    Path(out_path).write_text(f"""<!doctype html><meta charset=utf-8><title>{title}</title>
<style>body{{font:14px sans-serif;margin:12px}}canvas{{max-width:100%;border:1px solid #888;cursor:crosshair}}
button,select{{font-size:14px;margin-right:8px}}textarea{{width:100%;height:9em}}#m{{font-weight:bold;margin-left:8px}}</style>
<p>{intro}</p>
<select id=e{' hidden' if markers_only else ''}>{options}</select>
<button id=u>Undo</button><button id=s>Save {save_name}</button><button id=c>Copy JSON</button><span id=n></span><span id=m></span><br>
<canvas id=cv></canvas><textarea id=o readonly></textarea>
<script>
const ONLY_MARKERS={'true' if markers_only else 'false'};
const img=new Image();img.src={json.dumps(image_name)};const cv=document.getElementById('cv'),x=cv.getContext('2d');
const P={{front:[],right:[],back:[],left:[],markers:[]}},col={{front:'red',right:'lime',back:'cyan',left:'orange',markers:'magenta'}};
const say=t=>{{document.getElementById('m').textContent=t}};
const out=()=>JSON.stringify(ONLY_MARKERS?{{markers:P.markers}}:P);
img.onload=()=>{{cv.width=img.width;cv.height=img.height;draw()}};
img.onerror=()=>say('could not load '+img.src+': keep this page in the same folder as the frame');
function draw(){{x.drawImage(img,0,0);for(const k in P){{x.fillStyle=col[k];P[k].forEach(p=>{{const q=k=='markers'?p.px:p;x.beginPath();x.arc(q[0],q[1],4,0,7);x.fill();
if(k=='markers'){{x.font='bold 16px sans-serif';x.lineWidth=3;x.strokeStyle='black';const t=p.mm.join(',');x.strokeText(t,q[0]+7,q[1]-7);x.fillText(t,q[0]+7,q[1]-7)}}}})}}
document.getElementById('n').textContent=' '+Object.entries(P).filter(([k,v])=>!ONLY_MARKERS||k=='markers').map(([k,v])=>k+': '+v.length).join('  ');
document.getElementById('o').value=out()}}
cv.onclick=ev=>{{const r=cv.getBoundingClientRect(),s=cv.width/r.width;const pt=[(ev.clientX-r.left)*s,(ev.clientY-r.top)*s],k=document.getElementById('e').value;
if(k=='marker'){{const t=prompt('Bed X,Y in mm of this corner (e.g. 30,20)');if(t===null)return;
const m=(t.match(/-?\\d+(?:\\.\\d+)?/g)||[]).map(Number);if(m.length!=2){{say('not added: type two numbers, X then Y in mm, e.g. 30,20');return}}
P.markers.push({{px:pt,mm:m}});say('added X='+m[0]+', Y='+m[1])}}else{{P[k].push(pt);say('')}}draw()}};
document.getElementById('u').onclick=()=>{{const k=document.getElementById('e').value;P[k=='marker'?'markers':k].pop();say('');draw()}};
document.getElementById('s').onclick=()=>{{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out()],{{type:'application/json'}}));
a.download={json.dumps(save_name)};a.click();say('saved {save_name} (usually to Downloads)')}};
document.getElementById('c').onclick=()=>navigator.clipboard.writeText(out()).then(()=>say('copied'),()=>say('copy failed: use Save'));
</script>""")


def main(a):
    """`calibrate-bed`: (1) grab a frame + write the click page, (2) fit from points.json."""
    out = Path(a.out)
    work = out.with_suffix("")
    calib_z = getattr(a, "calib_z", None)
    if a.points and calib_z is None:
        raise SystemExit("--calib-z is required: the bed Z (mm, as the printer screen shows it after homing) "
                         "when the frame was grabbed. The P1S bed moves in Z, so the fit is only usable with it")
    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            raise SystemExit(f"cannot read {a.image}")
    else:
        from .printer.bambu import BambuCamera
        cam = BambuCamera(a.ip, a.code)
        frame = cam.grab()
        cam.close()
    h, w = frame.shape[:2]
    if not a.points:
        work.parent.mkdir(parents=True, exist_ok=True)
        png = work.with_name(work.name + "_frame.png")
        cv2.imwrite(str(png), frame)
        html = work.with_name(work.name + "_click.html")
        pick_points_html(png.name, html)
        print(f"frame saved: {png}  ({w}x{h})\nopen {html} in a browser, click 8 points along each bed edge "
              f"(plus the 4 corners), press Save points.json, then rerun with --points points.json "
              f"--calib-z <bed Z shown on the printer screen when this frame was grabbed>")
        return
    points = load_points(a.points)
    camera_height = getattr(a, "camera_height", None)
    if a.profile:
        lens = BedCamera.load(a.profile)
        cam_model, rep = refit_pose(lens, _corners_from_edges({e: points[e] for e in EDGES}), (a.bed, a.bed), calib_z)
        warn = cam_model.check_height(camera_height)
        rep.notes += [cam_model.pose_text()] + ([warn] if warn else [])
        rep.ok = rep.ok and not warn
    else:
        cam_model, rep = fit_from_bed_edges(points, (w, h), (a.bed, a.bed), a.fov, markers=load_markers(a.points),
                                            calib_z_mm=calib_z, camera_height_mm=camera_height,
                                            height_sigma_mm=getattr(a, "height_prior", None))
    print(rep.text())
    cv2.imwrite(str(work.with_name(work.name + "_overlay.png")), draw_overlay(cam_model, frame, (a.bed, a.bed)))
    if not rep.ok and not a.force:
        raise SystemExit("not saved: calibration is not acceptable (add more/better-spread clicks, or --force)")
    cam_model.save(out)
    print(f"saved {out}; check {work.name}_overlay.png: the green outline must sit on the plate edges")
