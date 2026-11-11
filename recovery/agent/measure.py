"""Where the agent's measurement of a change comes from during a dent test.

- ManualMeasure: a person types what the part now looks like ('dent X Y DEPTH', 'poke X Y',
  'none'; part-centred mm, e.g. the tray's front wall middle is 0 -10). Use it until the camera
  can measure the side-on P1S view: the decision, approval and action steps are the same.
- ScriptedMeasure: the same, given in advance (rehearsal and tests).
- CameraMeasure: the existing top-down measurement of the top layer (needs a top-down view).
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
