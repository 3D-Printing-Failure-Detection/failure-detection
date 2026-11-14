"""Where the agent's measurement of a change comes from during a dent test.

- ManualMeasure: a person types what the part now looks like ('dent X Y DEPTH', 'poke X Y',
  'none'; part-centred mm, e.g. the tray's front wall middle is 0 -10). The decision, approval
  and action steps are the same as with the camera.
- ScriptedMeasure: the same, given in advance (rehearsal and tests).
- SideViewMeasure: the printer camera (front-left, low, wide lens). The dent detector
  (recovery/models/dent.py) compares the frames saved before and after the change and says
  dent / poke / none, which wall and where along it; the person types the depth of a dent. Opt-in
  (--measure sideview) on the real printer until it is scored on real dent-test photos.
Each returns (new shapes for the top layers or None, a description for the log). A measure may set
`evidence` (a dict) for the approval prompt."""
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


DEPTH_PROMPT = ("The camera sees a DENT on the {wall} wall at ({x:.1f}, {y:.1f}) mm. How deep is it, in mm?\n"
                "  (decides resume vs correct: on the tray 1 to 1.5 mm still passes, 2 mm and more needs correct) > ")


def ask_depth(wall, x, y, ask=input, show=print):
    """The person types the depth of a dent the camera found (the detector does not measure depth)."""
    while True:
        a = ask(DEPTH_PROMPT.format(wall=wall, x=x, y=y)).strip()
        try:
            d = float(a)
            if d > 0:
                return d
        except ValueError:
            pass
        show("type the depth in mm, e.g. 1.5")


class SideViewMeasure:
    """Dent / poke / none from the before and after frames of the pause, in the P1S camera's own view.
    The camera gives the kind, the wall and the position; the detector does not measure depth, and depth
    decides resume vs correct, so `depth(wall, x, y)` supplies it: the person types it on the real printer,
    a rehearsal passes the scripted depth. A poke always opens the wall."""
    name = "sideview"

    def __init__(self, ckpt="runs/dent", cam=None, depth=ask_depth, taper_layers=5, yaw_deg=0.0):
        from ..models.dent import DentDetector
        self.det = DentDetector(ckpt, cam)
        self.depth, self.taper, self.yaw = depth, taper_layers, yaw_deg
        self.evidence = None

    def __call__(self, tools):
        before, after = tools.pause_frames
        top = tools.known[-self.taper:]
        origin = getattr(tools.pr, "origin", None) or tools.part.origin
        r = self.det.predict(before[..., ::-1], after[..., ::-1], origin, self.yaw, top[-1].z,
                             bounds=top[-1].region.bounds)
        min_area = self.det.cfg["min_area"]
        p_dent, p_poke = r["p_dent_vs_poke"]
        self.evidence = dict(kind=r["kind"], changed_px=r["change_px_crop"], min_px=min_area,
                             p_dent=p_dent, p_poke=p_poke, wall=r["wall"],
                             p_wall=r["wall_prob"].get(r["wall"]) if r["wall"] else None)
        note = f"camera (side view): {r['kind']}"
        if r["kind"] == "none" or r["at_mm"] is None:
            return None, note
        x, y = r["at_mm"]
        if r["kind"] == "dent":
            depth = float(self.depth(r["wall"], x, y))
            desc = f"dent {x:.1f} {y:.1f} {depth:g}"
        else:
            desc = f"poke {x:.1f} {y:.1f}"
        return change_from(desc, top), f"{note}, {r['wall']} wall -> '{desc}'" + (
            " (depth typed by the person)" if r["kind"] == "dent" and self.depth is ask_depth else "")


def describe_evidence(e):
    """One line on how sure the camera is, for the approval prompt."""
    if not e:
        return ""
    s = f"detector: {e['kind']}; changed area {e['changed_px']} px (needs {e['min_px']} px to count)"
    if e["kind"] != "none":
        s += f"; dent {e['p_dent']:.0%} / poke {e['p_poke']:.0%}; {e['wall']} wall {e['p_wall']:.0%}"
    if e["kind"] == "none" or e["changed_px"] < 2 * e["min_px"]:
        s += "  <- close to the threshold: check the images"
    return s
