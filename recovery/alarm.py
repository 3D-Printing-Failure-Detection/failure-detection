"""When a run of per-layer camera checks becomes an alarm. One rule, used by the agent, `watch` and the
whole-print test, so all three alarm on the same layer.

- The first `warmup` checked layers set this print's baseline: how much the camera always flags on it
  (calibration error, lighting, a shape the model reads slightly wrong). They never alarm.
- After that a layer counts as failing only if the detector says it is not ok AND it flags more than the
  baseline.
- An alarm needs `confirm` failing layers in a row: a real failure stays, a fluke does not.
"""
import numpy as np


class AlarmRule:
    def __init__(self, warmup=3, confirm=3):
        self.warmup, self.confirm = warmup, confirm
        self.seen, self.baseline, self.streak = [], 0.0, 0

    @property
    def warming_up(self):
        return len(self.seen) < self.warmup

    def update(self, label, flagged_px):
        """Feed one checked layer; True when this layer completes an alarm (and every layer after it that
        keeps failing). Unchecked layers should not be fed."""
        if self.warming_up:
            self.seen.append(flagged_px if label != "ok" else 0)
            self.baseline = float(np.median(self.seen))
            return False
        failing = label != "ok" and flagged_px > self.baseline
        self.streak = self.streak + 1 if failing else 0
        return self.streak >= self.confirm


def first_alarm(layers, warmup=3, confirm=3):
    """The first layer record ({kind, flagged_px, ...}) at which the rule alarms, or None."""
    rule = AlarmRule(warmup, confirm)
    for L in layers:
        if rule.update(L["kind"], L["flagged_px"]):
            return L
    return None
