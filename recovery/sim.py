"""Simulated printer: same interface as the real one (printer/bambu.py), so the
controller runs unchanged. The 'physics' is a surrogate: commanded layers are
deposited as planned, except (a) an active Failure displaces/thins them and
(b) once the controller adapts, the failure's cause is assumed transient and later
layers deposit as commanded (+ small positional noise)."""
from typing import List, Optional

import numpy as np
from shapely.affinity import translate

from . import camera
from .deform import Failure
from .types import Layer


class SimPrinter:
    def __init__(self, planned: List[Layer], failure: Optional[Failure] = None, seed=0,
                 start_idx=0, jitter_mm=0.06):
        self.nominal = planned
        self.plan = list(planned)
        self.failure = failure
        self.failure_active = failure is not None
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter_mm
        self.printed: List[Layer] = list(planned[:start_idx])      # clean layers before start
        self.paused = False
        self.cancelled = False
        self.adapted_from: Optional[int] = None

    # ----- printer interface -----
    @property
    def n_done(self):
        return len(self.printed)

    @property
    def n_total(self):
        return len(self.plan)

    def print_layer(self):
        if self.paused or self.cancelled or self.n_done >= self.n_total:
            raise RuntimeError("printer is paused, cancelled, or finished")
        idx = len(self.printed)
        lay = self.plan[idx]
        if self.failure_active and self.failure.active(idx):
            lay = self.failure.apply(lay, idx)
        elif self.adapted_from is not None and idx >= self.adapted_from and self.jitter > 0:
            d = self.rng.normal(0, self.jitter, 2)
            lay = Layer(lay.z, translate(lay.region, *d))
        self.printed.append(lay)
        return lay

    def camera_image(self):
        recent = [self.printed[0]] + self.printed[max(1, len(self.printed) - 4):]
        return camera.render(recent, self.rng)

    def expected_layer(self) -> Layer:
        return self.plan[len(self.printed) - 1]

    def pause(self):
        self.paused = True

    def replace_remaining(self, layers: List[Layer]):
        """Splice adapted layers in after what is already printed, and resume."""
        self.plan = self.plan[: len(self.printed)] + list(layers)
        self.adapted_from = len(self.printed)
        self.failure_active = False
        self.paused = False

    def resume(self):
        self.paused = False

    def cancel(self):
        self.cancelled = True

    def fast_forward(self):
        """Finish the print without imaging (used after the loop has been quiet)."""
        while len(self.printed) < len(self.plan) and not self.cancelled and not self.paused:
            self.print_layer()
