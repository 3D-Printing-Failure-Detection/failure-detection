"""The Bambu P1S chamber camera's view of a print, rendered in MuJoCo.

The top-down simulator (printer/mujoco.py) cannot show what the real camera sees: a fisheye lens low
at the front-left of the chamber, looking almost horizontally across the plate. This module renders
that view so that image models (e.g. dent detection) can be trained on synthetic frames and tested on
real photos.

Geometry (all mm, calibrated plate frame as in `camera.BedCamera`: origin at the plate's front-left
corner, +X right, +Y back, +Z up, with the plate where it was in the calibration frame):
- the camera is fixed; `P1SCamera` holds its intrinsics, k1/k2 lens distortion and pose;
- the P1S bed moves DOWN one layer per layer while the nozzle stays at the same height, so the top of
  the print is always at the nozzle plane, `calib_z_mm` above the calibrated plate (less `drop_mm`
  if the bed has gone further down, e.g. a Z-hop while paused). The plate itself is therefore at
  `calib_z_mm - top_z - drop_mm`: a taller print puts the plate lower in the image, not the print's top.

Rendering: a pinhole image wide enough to cover the lens is rendered by MuJoCo, then resampled through
the OpenCV distortion model into the 1280x720 camera image (`remap_maps`). A depth image is resampled
the same way, so before/after depth differences give a pixel mask of what changed.

The printed part is a triangle mesh built from the layer stack (`stack_mesh`): every layer is a band
with a slight outward bulge at mid-height, which shows the layer lines under side lighting, plus the
exposed top surface of each layer.
"""
import os
import sys
from dataclasses import dataclass, field, replace
from typing import List, Optional

if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):   # headless: GPU via EGL
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np
import shapely
from shapely.geometry.polygon import orient

from .geometry.ops import clean, polys
from .types import Layer

W, H = 1280, 720                       # P1S chamber camera stream
BED_MM = 256.0
CAMERA_ABOVE_PRINT_PLANE_MM = 22.784   # measured by the team (calibration-changes branch, config.py)

# Default camera: fitted with recovery/calib_offline.py (calibration-changes branch) to the real P1S
# bed-edge clicks in tests/data/p1s_clicks_issue5.json, calib_z = 5 mm, with the measured camera height
# as a 1 mm prior (the only calib_z for which the fitted height agrees with it). The fit's edge residual
# is 4.5 mm RMS, above the 1 mm acceptance threshold, so treat it as approximate: the dataset generator
# jitters it, and a proper calibration file should replace it (`P1SCamera.load`).
DEFAULT_CAMERA = dict(f=800.984, dist=[-0.304012, 0.099058, 0.0, 0.0, 0.0],
                      rvec=[1.566748, -0.870601, 0.751848], tvec=[46.3079, 25.6619, 13.5881], calib_z_mm=5.0)


@dataclass
class P1SCamera:
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    calib_z_mm: float
    size: tuple = (W, H)

    @classmethod
    def default(cls) -> "P1SCamera":
        d = DEFAULT_CAMERA
        K = np.array([[d["f"], 0, W / 2], [0, d["f"], H / 2], [0, 0, 1.0]])
        return cls(K, np.array(d["dist"], float), np.array(d["rvec"], float).reshape(3, 1),
                   np.array(d["tvec"], float).reshape(3, 1), d["calib_z_mm"])

    @classmethod
    def load(cls, path) -> "P1SCamera":
        """A calibration saved by `BedCamera.save` (calibrate-bed / calibrate-board). It must record the
        bed Z of its frame (calib_z_mm): without it the moving bed cannot be placed."""
        d = np.load(path)
        if "calib_z_mm" not in d.files:
            raise ValueError(f"{path} has no calib_z_mm (bed Z of the calibration frame); recalibrate with --calib-z")
        return cls(d["K"].astype(float), d["dist"].astype(float).ravel(), d["rvec"].astype(float).reshape(3, 1),
                   d["tvec"].astype(float).reshape(3, 1), float(d["calib_z_mm"]))

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ np.asarray(self.tvec, float).reshape(3)

    def project(self, pts) -> np.ndarray:
        px, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 3), self.rvec, self.tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    def jittered(self, rng, pos_mm=2.0, rot_deg=0.8, f_rel=0.02, k1=0.02, k2=0.01) -> "P1SCamera":
        """Same camera with small random errors: covers calibration uncertainty and unit-to-unit spread."""
        c = self.centre + rng.normal(0, pos_mm, 3)
        R = cv2.Rodrigues(np.radians(rng.normal(0, rot_deg, 3)))[0] @ self.R
        K = self.K.copy()
        K[0, 0] = K[1, 1] = K[0, 0] * (1 + rng.normal(0, f_rel))
        dist = self.dist.copy()
        dist[0] += rng.normal(0, k1)
        dist[1] += rng.normal(0, k2)
        return replace(self, K=K, dist=dist, rvec=cv2.Rodrigues(R)[0], tvec=(-R @ c).reshape(3, 1))

    def to_json(self) -> dict:
        return dict(K=self.K.round(4).tolist(), dist=self.dist.round(6).tolist(),
                    rvec=np.ravel(self.rvec).round(6).tolist(), tvec=np.ravel(self.tvec).round(4).tolist(),
                    calib_z_mm=self.calib_z_mm, centre_mm=self.centre.round(2).tolist())


def _distort(x, y, dist):
    k1, k2, p1, p2, k3 = (list(np.ravel(dist)) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def _undistort(px, K, dist, iters=300, tol_px=0.01):
    """Pixels (...,2) of the distorted image -> normalized pinhole coordinates (...,2). Same model as
    OpenCV, inverted by damped fixed-point iteration and checked by re-distorting."""
    xd = (px[..., 0] - K[0, 2]) / K[0, 0]
    yd = (px[..., 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        fx, fy = _distort(x, y, dist)
        x, y = x + 0.7 * (xd - fx), y + 0.7 * (yd - fy)
    fx, fy = _distort(x, y, dist)
    err = np.hypot((fx - xd) * K[0, 0], (fy - yd) * K[1, 1])
    if not np.all(err < tol_px):
        raise ValueError(f"lens distortion cannot be inverted over the image (max error {np.nanmax(err):.2f} px)")
    return np.stack([x, y], -1)


def _grid_sample(coarse, step, w, h):
    """Bilinear interpolation of values on the grid (0, step, 2*step, ...) at every integer pixel."""
    gx = (np.arange(w, dtype=np.float32) / step)[None, :].repeat(h, 0)
    gy = (np.arange(h, dtype=np.float32) / step)[:, None].repeat(w, 1)
    return np.stack([cv2.remap(coarse[..., k], gx, gy, cv2.INTER_LINEAR) for k in range(2)], -1).astype(np.float64)


def remap_maps(cam: P1SCamera, margin=1.02):
    """Pinhole render size/focal and the maps that resample it into the distorted camera image.
    Returns (map_x, map_y, render_w, render_h, render_f)."""
    w, h = cam.size
    step = 8                                     # the maps are smooth: invert on a grid, interpolate
    us, vs = np.arange(0, w + step, step, dtype=np.float64), np.arange(0, h + step, step, dtype=np.float64)
    coarse = _undistort(np.stack(np.meshgrid(us, vs), -1), cam.K, cam.dist).astype(np.float32)
    xy = _grid_sample(coarse, step, w, h)
    f = float(cam.K[0, 0])
    hx, hy = np.abs(xy[..., 0]).max() * margin, np.abs(xy[..., 1]).max() * margin
    if not (np.isfinite(hx) and np.isfinite(hy)) or hx > 4 or hy > 4:
        raise ValueError("lens distortion cannot be inverted over the image (|k1| too large?)")
    rw, rh = 2 * int(np.ceil(f * hx)) + 2, 2 * int(np.ceil(f * hy)) + 2
    map_x = (rw / 2 - 0.5 + f * xy[..., 0]).astype(np.float32)
    map_y = (rh / 2 - 0.5 + f * xy[..., 1]).astype(np.float32)
    return map_x, map_y, rw, rh, f


# ---------------------------------------------------------------- part mesh
def _ring_band(ring_xy, z_levels, offsets):
    """Side wall of one ring (material on the left, i.e. shapely orient(sign=1)) through the given z levels,
    each level pushed `offset` mm away from the material. Returns verts (N,3), faces (M,3) (outward)."""
    p = np.asarray(ring_xy, float)[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
    p = p[keep]
    if len(p) >= 2 and np.linalg.norm(p[0] - p[-1]) < 1e-6:
        p = p[:-1]
    n = len(p)
    if n < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    d = np.roll(p, -1, 0) - p
    en = np.stack([d[:, 1], -d[:, 0]], 1)
    en /= np.linalg.norm(en, axis=1, keepdims=True)
    vn = en + np.roll(en, 1, 0)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    verts = np.concatenate([np.c_[p + o * vn, np.full(n, z)] for z, o in zip(z_levels, offsets)])
    faces = []
    j = np.arange(n)
    for lv in range(len(z_levels) - 1):
        a0, b0 = lv * n + j, lv * n + (j + 1) % n
        a1, b1 = a0 + n, b0 + n
        faces += [np.stack([a0, b0, b1], 1), np.stack([a0, b1, a1], 1)]
    return verts, np.concatenate(faces)


def _cap(region, z):
    """Upward-facing triangles covering `region` at height z."""
    vs, fs, k = [], [], 0
    for p in polys(region):
        tris = shapely.constrained_delaunay_triangles(p)
        for t in getattr(tris, "geoms", []):
            c = np.asarray(orient(t, 1.0).exterior.coords)[:3]
            vs.append(np.c_[c, np.full(3, z)])
            fs.append([k, k + 1, k + 2])
            k += 3
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(vs), np.array(fs)


def stack_mesh(layers: List[Layer], lo: int = 0, hi: Optional[int] = None, bulge_mm=0.06,
               z_bottom_of_first=0.0):
    """Mesh of layers[lo:hi] (part frame, mm). Each layer: side walls of every ring from the previous
    layer's z to its own, bulging `bulge_mm` outward at mid-height, plus the top surface not covered
    by the next layer. Returns verts (N,3), faces (M,3)."""
    hi = len(layers) if hi is None else hi
    V, F, n = [], [], 0

    def add(v, f):
        nonlocal n
        if len(v):
            V.append(v)
            F.append(f + n)
            n += len(v)

    for i in range(lo, hi):
        lay = layers[i]
        zb = z_bottom_of_first if i == 0 else layers[i - 1].z
        zt = lay.z
        for p in polys(lay.region):
            p = orient(p, 1.0)
            for ring in [p.exterior, *p.interiors]:
                add(*_ring_band(ring.coords, (zb, (zb + zt) / 2, zt), (0.0, bulge_mm, 0.0)))
        above = layers[i + 1].region if i + 1 < len(layers) else None
        top = lay.region if above is None else clean(lay.region.difference(above), min_area=0.02)
        add(*_cap(top, zt))
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(V), np.concatenate(F)


# ---------------------------------------------------------------- scene
@dataclass
class Look:
    """Everything about appearance that the generator randomises."""
    filament_rgb: tuple = (0.9, 0.9, 0.88)
    filament_specular: float = 0.3
    filament_shininess: float = 0.4
    plate_rgb1: tuple = (0.45, 0.36, 0.20)       # textured PEI: gold-brown speckle
    plate_rgb2: tuple = (0.30, 0.24, 0.13)
    plate_specular: float = 0.2
    chamber_rgb: tuple = (0.12, 0.12, 0.13)
    ambient: float = 0.25
    headlight: float = 0.15
    lights: list = field(default_factory=lambda: [
        dict(pos=(60.0, 40.0, 180.0), target=(60.0, 60.0, 0.0), diffuse=0.7, specular=0.2, spot=True, shadow=False)])
    toolhead_xy: Optional[tuple] = None          # nozzle position on the bed plane (mm); None = out of view
    toolhead_rgb: tuple = (0.15, 0.15, 0.16)


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def _scene_xml(cam: P1SCamera, look: Look, plate_z, rw, rh, rf):
    m = 1e-3
    R = cam.R
    xaxis, yaxis = R[0], -R[1]                   # OpenCV (right, down, forward) -> MuJoCo (right, up, back)
    fovy = np.degrees(2 * np.arctan(rh / 2 / rf))
    nozzle_z = cam.calib_z_mm
    lights = []
    # shadow maps alias on the 0.2 mm layer ridges (checkerboard acne at this scale), so lights default to no
    # shadows; a dent shows through the shading of the wall, not through cast shadows
    for L in look.lights:
        pos, tgt = np.array(L["pos"], float), np.array(L["target"], float)
        d = (tgt - pos) / np.linalg.norm(tgt - pos)
        lights.append(f'<light pos="{_f(pos * m)}" dir="{_f(d)}" diffuse="{_f([L["diffuse"]] * 3)}" '
                      f'specular="{_f([L["specular"]] * 3)}" castshadow="{"true" if L.get("shadow") else "false"}" '
                      f'directional="{"false" if L.get("spot", True) else "true"}" cutoff="60" exponent="2" '
                      f'attenuation="1 0 0"/>')
    tool = ""
    if look.toolhead_xy is not None:
        tx, ty = look.toolhead_xy
        # P1S toolhead: roughly 50 x 45 mm, nozzle tip at the nozzle plane, body from ~4 mm above it
        tool = (f'<geom type="cylinder" size="{.6 * m} {2 * m}" pos="{_f([tx * m, ty * m, (nozzle_z + 2) * m])}" '
                f'rgba=".75 .6 .25 1"/>'
                f'<geom type="box" size="{25 * m} {22 * m} {35 * m}" pos="{_f([tx * m, (ty + 8) * m, (nozzle_z + 39) * m])}" '
                f'rgba="{_f(look.toolhead_rgb)} 1"/>')
    return f'''<mujoco model="p1s_view">
  <visual>
    <global offwidth="{rw}" offheight="{rh}"/>
    <headlight ambient="{_f([look.ambient] * 3)}" diffuse="{_f([look.headlight] * 3)}" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.01" zfar="3" shadowclip="1.5"/>
  </visual>
  <statistic extent="0.3" center="0.128 0.128 0"/>
  <asset>
    <texture name="pei" type="2d" builtin="flat" mark="random" random="0.45" width="512" height="512"
             rgb1="{_f(look.plate_rgb1)}" rgb2="{_f(look.plate_rgb2)}" markrgb="{_f(look.plate_rgb2)}"/>
    <material name="pei" texture="pei" texrepeat="40 40" specular="{look.plate_specular:.3g}" shininess="0.3"/>
    <material name="fil" rgba="{_f(look.filament_rgb)} 1" specular="{look.filament_specular:.3g}"
              shininess="{look.filament_shininess:.3g}"/>
  </asset>
  <!-- every geom is static in the worldbody, so nothing collides; no contype defaults (with convex mesh
       inertia they crash MuJoCo 3.2.7) -->
  <worldbody>
    <camera name="cam" pos="{_f(cam.centre * m)}" xyaxes="{_f(np.r_[xaxis, yaxis])}" fovy="{fovy:.6f}"/>
    {"".join(lights)}
    <geom name="plate" type="box" material="pei" size="{BED_MM / 2 * m} {BED_MM / 2 * m} {0.5 * m}"
          pos="{_f([BED_MM / 2 * m, BED_MM / 2 * m, (plate_z - 0.5) * m])}"/>
    <geom type="box" size="{(BED_MM / 2 + 15) * m} {(BED_MM / 2 + 15) * m} {4 * m}" rgba=".05 .05 .05 1"
          pos="{_f([BED_MM / 2 * m, BED_MM / 2 * m, (plate_z - 5) * m])}"/>
    <geom type="box" size="{200 * m} {2 * m} {200 * m}" rgba="{_f(look.chamber_rgb)} 1" pos="{_f([128 * m, 300 * m, 0])}"/>
    <geom type="box" size="{2 * m} {200 * m} {200 * m}" rgba="{_f(look.chamber_rgb)} 1" pos="{_f([300 * m, 128 * m, 0])}"/>
    <geom type="box" size="{200 * m} {200 * m} {2 * m}" rgba="{_f(np.array(look.chamber_rgb) * .7)} 1"
          pos="{_f([128 * m, 128 * m, (plate_z - 150) * m])}"/>
    {tool}
  </worldbody>
</mujoco>'''


def _part_world(verts, xy, yaw_deg, plate_z):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    out = verts.copy()
    out[:, 0] = c * verts[:, 0] - s * verts[:, 1] + xy[0]
    out[:, 1] = s * verts[:, 0] + c * verts[:, 1] + xy[1]
    out[:, 2] = verts[:, 2] + plate_z
    return out


def plate_z_mm(cam: P1SCamera, top_z, drop_mm=0.0):
    """Height of the plate surface (calibrated frame) when the print's top `top_z` is at the nozzle plane,
    with the bed a further `drop_mm` down."""
    return cam.calib_z_mm - top_z - drop_mm


class P1SScene:
    """One scene: fixed camera, look and plate height, with several versions of the printed part that can be
    rendered one at a time (e.g. 'before' and 'after' a dent). Mesh `parts` = {name: (verts, faces)} in the
    part frame; a version is a list of part names to show together.

        with P1SScene(cam, look, {"base": m0, "top0": m1, "top1": m2}, xy=(80, 90), yaw_deg=0,
                      top_z=6.0) as s:
            rgb0, depth0 = s.render(["base", "top0"])
    """

    def __init__(self, cam: P1SCamera, look: Look, parts: dict, xy, yaw_deg, top_z, drop_mm=0.0):
        self.cam, self.look = cam, look
        self.plate_z = plate_z_mm(cam, top_z, drop_mm)
        self.map_x, self.map_y, rw, rh, rf = remap_maps(cam)
        assets, geoms, self.groups = [], [], {}
        for gi, (name, (v, f)) in enumerate(parts.items()):
            if not len(f):
                self.groups[name] = None
                continue
            if gi >= 5:
                raise ValueError("at most 5 part meshes per scene (geom groups 1..5)")
            vw = _part_world(v, xy, yaw_deg, self.plate_z) * 1e-3
            # the stack is an open surface: convex-hull inertia keeps the mesh where it was put (the default
            # and shell inertia shift an open mesh in MuJoCo 3.2; MjSpec crashes when it is set there)
            assets.append(f'<mesh name="part_{name}" inertia="convex" smoothnormal="false" '
                          f'vertex="{" ".join(f"{x:.7g}" for x in vw.ravel())}" '
                          f'face="{" ".join(map(str, np.asarray(f, int).ravel()))}"/>')
            geoms.append(f'<geom type="mesh" mesh="part_{name}" material="fil" group="{1 + gi}"/>')
            self.groups[name] = 1 + gi
        xml = _scene_xml(cam, look, self.plate_z, rw, rh, rf)
        xml = xml.replace("</asset>", "".join(assets) + "</asset>").replace("</worldbody>", "".join(geoms) + "</worldbody>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, rh, rw)

    def render(self, show: List[str], scene=True):
        """(rgb uint8 HxWx3 in the distorted camera image, depth in mm along the optical axis).
        scene=False hides the plate, chamber and toolhead: only the named part meshes are drawn."""
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = int(scene)
        for name in show:
            if self.groups.get(name):
                opt.geomgroup[self.groups[name]] = 1
        self.renderer.update_scene(self.data, camera="cam", scene_option=opt)
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render() * 1000.0
        self.renderer.disable_depth_rendering()
        rgb = cv2.remap(rgb, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        depth = cv2.remap(depth.astype(np.float32), self.map_x, self.map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)
        return rgb, depth

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- camera noise
@dataclass
class Photometric:
    """Sensor/ISP effects applied after rendering (per pair; `perturbed` for the second frame)."""
    gain: float = 1.0
    wb: tuple = (1.0, 1.0, 1.0)                  # per-channel gains (RGB)
    gamma: float = 1.0
    vignette: float = 0.25
    blur_sigma: float = 0.6
    noise_std: float = 3.0                       # DN, signal-independent
    shot_noise: float = 0.6                      # scales sqrt(signal)
    jpeg_quality: int = 70

    @classmethod
    def sample(cls, rng):
        return cls(gain=rng.uniform(0.7, 1.35), wb=tuple(rng.uniform(0.85, 1.15, 3)), gamma=rng.uniform(0.8, 1.25),
                   vignette=rng.uniform(0.0, 0.5), blur_sigma=rng.uniform(0.3, 1.3), noise_std=rng.uniform(0.5, 6.0),
                   shot_noise=rng.uniform(0.0, 1.2), jpeg_quality=int(rng.integers(35, 92)))

    def perturbed(self, rng, gain_rel=0.04):
        return replace(self, gain=self.gain * (1 + rng.normal(0, gain_rel)))

    def apply(self, rgb, rng, jpeg=True):
        """Camera effects on an RGB render. jpeg=False leaves the compression to whoever saves the frame
        (with `jpeg_quality`), so it is not applied twice."""
        h, w = rgb.shape[:2]
        x = rgb.astype(np.float32) / 255.0
        x = x * np.array(self.wb, np.float32) * self.gain
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        x *= (1 - self.vignette * r2)[..., None]
        x = np.clip(x, 0, 1) ** self.gamma
        if self.blur_sigma > 0.05:
            x = cv2.GaussianBlur(x, (0, 0), self.blur_sigma)
        x = x * 255 + rng.normal(0, 1, x.shape).astype(np.float32) * (
            self.noise_std + self.shot_noise * np.sqrt(np.maximum(x * 255, 0)))
        out = np.clip(x, 0, 255).astype(np.uint8)
        if not jpeg:
            return out
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
