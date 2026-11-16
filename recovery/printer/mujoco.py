"""Hybrid MuJoCo printer: actuated stages + rendered RGB + geometric deposition.

The OVERHEAD RESEARCH RIG: its camera looks straight down, which the real P1S cannot. It is kept for the
report's results (docs/mujoco-report.md) and their tests. The real printer and its rehearsal sim use only the
P1S camera's own view (recovery/p1s_view.py, recovery/agent/rehearse.py).

Every generated extrusion polyline is executed as settled waypoint moves. The
realized nozzle/bed coordinates are buffered into beads, then a prescribed failure
deforms the deposited layer. This is a quasi-static toolpath model: no melt,
thermal contraction, adhesion, belt compliance, or feed-rate dynamics are claimed.
The camera is a MuJoCo perspective render, rectified using its known calibration.
No segmentation IDs or true contour are passed to the controller.
"""
import os
import sys

if sys.platform.startswith("linux"):          # headless servers; macOS/Windows use their native GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import shapely

from .. import camera
from ..gcode import layer_paths, write_layers
from ..geometry.ops import clean
from ..sim import SimPrinter
from ..types import Layer
from .profiles import PROFILES


def model_xml(profile):
    p = profile
    bx, by = np.array(p.bed_mm) / 2000
    # Diagonal coordinates implement the ideal CoreXY motor-to-XY transform.
    axes = (".707106781 .707106781 0", ".707106781 -.707106781 0") if p.kinematics == "corexy" else ("1 0 0", "0 1 0")
    yjoint = '' if p.kinematics == "bedslinger" else f'<joint name="v" axis="{axes[1]}"/>'
    bedjoint = '<joint name="v" axis="0 1 0"/>' if p.kinematics == "bedslinger" else ''
    pos = np.array(p.camera_position_m)
    zaxis = pos - np.array([0, 0, .025])
    zaxis /= np.linalg.norm(zaxis)
    xaxis = np.cross([0, 1, 0], zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    xyaxes = ' '.join(map(str, np.r_[xaxis, yaxis]))
    cam_pos = ' '.join(map(str, pos))
    return f'''<mujoco model="{p.name}">
      <option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
      <visual><global offwidth="640" offheight="640"/>
        <headlight ambient=".5 .5 .5" diffuse=".5 .5 .5" specular="0 0 0"/>
        <map znear=".001" zfar="5"/></visual>
      <default><joint type="slide" damping="100"/>
        <geom contype="0" conaffinity="0" friction="0 0 0"/>
        <position kp="40000" kv="100"/></default>
      <worldbody>
        <camera name="inspection" pos="{cam_pos}" xyaxes="{xyaxes}" fovy="36"/>
        <geom type="box" pos="0 0 -.018" size="{bx+.012} {by+.012} .008" rgba=".07 .08 .09 1"/>
        <body name="bed">{bedjoint}
          <geom type="box" pos="0 0 -.004" size="{bx} {by} .004" mass="{p.moving_mass_kg}" rgba=".18 .19 .2 1"/>
          <site name="bed_origin" size=".001" rgba="0 0 0 0"/>
        </body>
        <body name="head"><joint name="u" axis="{axes[0]}"/>{yjoint}
          <joint name="z" axis="0 0 1"/>
          <geom type="box" pos="0 0 .008" size=".004 .004 .008" mass="{p.moving_mass_kg}" rgba=".2 .22 .25 1"/>
          <site name="nozzle" size=".001" rgba="0 0 0 0"/>
        </body>
      </worldbody>
      <actuator><position joint="u"/><position joint="v"/><position joint="z"/></actuator>
    </mujoco>'''


class MujocoPrinter(SimPrinter):
    def __init__(self, planned, failure=None, seed=0, profile="cartesian",
                 clear_failure_on_adapt=False, render=True):
        super().__init__(planned, failure, seed=seed, jitter_mm=0)
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.clear_failure_on_adapt = clear_failure_on_adapt
        self.validate_plan(planned)
        self.model = mujoco.MjModel.from_xml_string(model_xml(self.profile))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, 640, 640, max_geom=20000) if render else None
        self.rng_camera = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.executed_waypoints = 0
        self.tracking_squared_mm = 0.0
        self.max_tracking_mm = 0.0
        self.camera_frames = 0
        self.replacements = []
        self._paths_cache = {}
        self._calibration = self._camera_calibration()

    def _camera_calibration(self):
        cid = self.model.camera("inspection").id
        # MuJoCo camera: right/up/back; OpenCV: right/down/forward.
        R = np.diag([1., -1., -1.]) @ self.data.cam_xmat[cid].reshape(3, 3).T
        t = -R @ (self.data.cam_xpos[cid] * 1000)
        f = 320 / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        K = np.array([[f, 0, 319.5], [0, f, 319.5], [0, 0, 1]])
        return camera.BedCamera(K, np.zeros(5), cv2.Rodrigues(R)[0], t)

    def validate_plan(self, layers):
        bx, by = np.array(self.profile.bed_mm) / 2
        last = 0
        for lay in layers:
            if not np.isfinite(lay.z) or not last < lay.z <= self.profile.height_mm:
                raise ValueError("layer heights must increase within the printer build volume")
            last = lay.z
            if lay.region.is_empty:
                continue
            x0, y0, x1, y1 = lay.region.bounds
            if x0 < -bx or x1 > bx or y0 < -by or y1 > by:
                raise ValueError("toolpath exceeds the printer bed")
            if max(abs(v) for v in lay.region.bounds) > camera.CAM.fov_mm / 2 - 2:
                raise ValueError("toolpath exceeds the calibrated inspection field of view")

    def move(self, xyz_mm):
        xyz = np.asarray(xyz_mm, float) / 1000
        x, y, z = xyz
        if self.profile.kinematics == "corexy":
            u, v = (x + y) / np.sqrt(2), (x - y) / np.sqrt(2)
        elif self.profile.kinematics == "bedslinger":
            u, v = x, -y
        else:
            u, v = x, y
        self.data.ctrl[:] = [u, v, z]
        mujoco.mj_step(self.model, self.data, nstep=40)
        mujoco.mj_forward(self.model, self.data)
        actual = (self.data.site("nozzle").xpos - self.data.site("bed_origin").xpos) * 1000
        error = float(np.linalg.norm(actual - xyz_mm))
        self.executed_waypoints += 1
        self.tracking_squared_mm += error ** 2
        self.max_tracking_mm = max(self.max_tracking_mm, error)
        return actual

    @property
    def tracking_rms_mm(self):
        return np.sqrt(self.tracking_squared_mm / max(1, self.executed_waypoints))

    def print_layer(self):
        if self.paused or self.cancelled or self.n_done >= self.n_total:
            raise RuntimeError("printer is paused, cancelled, or finished")
        k = self.n_done
        lay = self.plan[k]
        key = (lay.region.wkb, k % 2)
        if key not in self._paths_cache:
            self._paths_cache[key] = layer_paths(lay.region, k, self.profile.line_width_mm)
        traces = []
        for path in self._paths_cache[key]:
            realized = [self.move([xy[0], xy[1], lay.z])[:2] for xy in path]
            if len(realized) > 1:
                traces.append(shapely.LineString(realized))
        reg = clean(shapely.union_all(shapely.buffer(traces, self.profile.line_width_mm / 2,
                                                    quad_segs=2))) if traces else shapely.Polygon()
        # Continuum deposition assumes adjacent beads fuse across gaps up to
        # 0.2 mm. Otherwise polygonal circle/cap approximations leave many tiny
        # pores that invalidate a continuum wall-thickness check. This explicit
        # model assumption is applied BEFORE failure injection, so failed walls
        # and open gaps are not repaired by the simulator.
        reg = clean(reg.buffer(.1).buffer(-.1).simplify(.005))
        actual = Layer(lay.z, reg)
        if self.failure_active and self.failure.active(k):
            actual = self.failure.apply(actual, k)
        self.printed.append(actual)
        return actual

    def replace_remaining(self, layers):
        if not self.paused:
            raise RuntimeError("pause before replacing the remaining layers")
        if len(layers) != self.n_total - self.n_done:
            raise ValueError("replacement must preserve the remaining layer count")
        self.validate_plan(self.plan[:self.n_done] + list(layers))
        if any(a.z != b.z for a, b in zip(layers, self.plan[self.n_done:])):
            raise ValueError("replacement must preserve the remaining layer heights")
        h = self.plan[0].z if len(self.plan) == 1 else self.plan[1].z - self.plan[0].z
        gc, _ = write_layers(layers, h, start_idx=self.n_done, w=self.profile.line_width_mm,
                             origin=tuple(np.array(self.profile.bed_mm) / 2))
        self.replacements.append(gc)
        self.plan = self.plan[:self.n_done] + list(layers)
        self.adapted_from = self.n_done
        if self.clear_failure_on_adapt:
            self.failure_active = False
        self.paused = False

    def _add_layer_visual(self, layer, color, thickness):
        # Close numerical cracks far below the camera's spatial resolution.
        # The physical truth and function checks retain the original geometry.
        cfg = camera.C.CameraCfg(px_per_mm=4, fov_mm=camera.CAM.fov_mm)
        visible = layer.region.buffer(.02).buffer(-.02)
        mask = camera.rasterise(visible, cfg) > 127
        scene = self.renderer.scene
        for row in range(mask.shape[0]):
            ends = np.flatnonzero(np.diff(np.r_[False, mask[row], False].astype(np.int8)))
            for start, stop in ends.reshape(-1, 2):
                if scene.ngeom >= scene.maxgeom:
                    raise RuntimeError("MuJoCo scene capacity exceeded")
                xy = camera.px_to_mm(np.array([[(start + stop - 1) / 2, row]]), cfg)[0]
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([(stop-start) / cfg.px_per_mm / 2000, .5 / cfg.px_per_mm / 1000, thickness / 2000]),
                    np.r_[xy / 1000, (layer.z - thickness/2) / 1000], np.eye(3).ravel(),
                    np.r_[color, 1].astype(np.float32))
                scene.ngeom += 1

    def camera_image(self):
        if self.renderer is None:
            raise RuntimeError("renderer was disabled")
        lay = self.printed[-1]
        # Park outside the inspected crop and return a moving bed to its datum.
        self.move([self.profile.bed_mm[0] / 2 - 8, 0, lay.z + 15])
        self.renderer.update_scene(self.data, camera="inspection")
        color = np.array(self.profile.filament_rgb)
        # Controlled inspection illumination: older material is darker. This is
        # an explicit optical assumption, not a thermal/plastic material model.
        if len(self.printed) > 1:
            self._add_layer_visual(self.printed[0], color * .40, self.printed[0].z)
            for old in self.printed[max(1, len(self.printed)-3):-1]:
                self._add_layer_visual(old, color * .60, .25)
        self._add_layer_visual(lay, color, .25)
        rgb = self.renderer.render()
        bgr = rgb[..., ::-1].copy()
        bgr = np.clip(bgr.astype(float) + self.rng_camera.normal(0, self.profile.noise_std, bgr.shape),
                      0, 255).astype(np.uint8)
        self.last_raw_image = bgr
        self.camera_frames += 1
        return self._calibration.top_down(bgr, lay.z, (0, 0))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
