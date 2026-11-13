"""Where the agent's measurement of a change comes from during a dent test.

- ManualMeasure: a person types what the part now looks like ('dent X Y DEPTH', 'poke X Y',
  'none'; part-centred mm, e.g. the tray's front wall middle is 0 -10). Use it until the camera
  can measure the side-on P1S view: the decision, approval and action steps are the same.
- ScriptedMeasure: the same, given in advance (rehearsal and tests).
- SideViewMeasure: the P1S chamber camera's own view (front-left, near-horizontal fisheye). The dent
  detector (recovery/models/dent.py) compares the frames saved before and after the change and
  says dent / poke / none, which wall and where along it. The default.
- CameraMeasure: the top-down measurement of the top layer. The P1S camera is too shallow for a
  usable top-down view (2 to 4 mm per pixel mid-plate), so this is for other rigs and the old sim.
Each returns (new shapes for the top layers or None, a description for the log)."""
from .rehearse import change_from

PROMPT = ("Describe the part now (mm from the part centre): 'dent X Y DEPTH', 'poke X Y' or 'none'\n"
          "  tray: front wall middle = 0 -10, right wall middle = 15 0 > ")


class ManualMeasure:
    name = "manual"

    def __init__(self, ask=input, show=print, taper_layers=5):
        self.ask, self.show, self.taper = ask, show, taper_layers

    def __call__(self, tools):
        top = tools.known[-self.taper:]
        while True:
            desc = self.ask(PROMPT).strip() or "none"
            try:
                return change_from(desc, top), desc
            except ValueError as e:
                self.show(e)


class ScriptedMeasure:
    name = "scripted"

    def __init__(self, description, taper_layers=5):
        self.description, self.taper = description, taper_layers

    def __call__(self, tools):
        return change_from(self.description, tools.known[-self.taper:]), self.description


class CameraMeasure:
    name = "camera"

    def __call__(self, tools):
        obs = tools.observe()
        return ([obs.measured] if obs.label != "ok" else None), f"camera: {obs.label}, {obs.deviation_mm:.2f} mm"


class SideViewMeasure:
    """Dent / poke / none from the before and after frames of the pause, in the P1S camera's own view.
    The detector does not measure depth: a dent is taken as `dent_depth_mm` deep (the test's gentle push
    is 1 to 2 mm), which only matters for the capacity check; a poke always opens the wall."""
    name = "sideview"

    def __init__(self, ckpt="runs/dent", cam=None, dent_depth_mm=1.5, taper_layers=5, yaw_deg=0.0):
        from ..models.dent import DentDetector
        self.det = DentDetector(ckpt, cam)
        self.depth, self.taper, self.yaw = dent_depth_mm, taper_layers, yaw_deg

    def __call__(self, tools):
        before, after = tools.pause_frames
        top = tools.known[-self.taper:]
        origin = getattr(tools.pr, "origin", None) or tools.part.origin
        r = self.det.predict(before[..., ::-1], after[..., ::-1], origin, self.yaw, top[-1].z,
                             bounds=top[-1].region.bounds)
        note = f"camera (side view): {r['kind']}"
        if r["kind"] == "none" or r["at_mm"] is None:
            return None, note
        x, y = r["at_mm"]
        desc = f"dent {x:.1f} {y:.1f} {self.depth}" if r["kind"] == "dent" else f"poke {x:.1f} {y:.1f}"
        return change_from(desc, top), f"{note}, {r['wall']} wall -> '{desc}' (depth not measured)"
