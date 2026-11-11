import os
import sys
if sys.platform.startswith("linux"):          # headless servers; macOS/Windows use their native GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
from shapely.geometry import Point, box

pytest.importorskip("mujoco")

from recovery.deform import random_failure
from recovery.geometry.ops import iou
from recovery.loop.controller import Controller
from recovery.loop.decide import RulePolicy
from recovery.models.classical import ClassicalDetector, ClassicalSegmenter
from recovery.printer.mujoco import MujocoPrinter
from recovery.printer.profiles import PROFILES
from recovery.types import Layer


@pytest.mark.parametrize("profile", PROFILES)
def test_rig_motion_camera_and_full_controller(profile):
    region = Point(0, 0).buffer(10).difference(Point(0, 0).buffer(5))
    layers = [Layer(.25*(i+1), region) for i in range(4)]
    spec = {"material_volume_mm3": [200, 260]}
    with MujocoPrinter(layers, profile=profile) as pr:
        actual = pr.move([13, -8, 2])
        assert np.allclose(actual, [13, -8, 2], atol=.001)
        result = Controller(pr, ClassicalSegmenter(), ClassicalDetector(),
                            RulePolicy(spec, layers, .25), spec, layers, .25).run()
        assert result.functional and pr.camera_frames == len(layers)
        assert not result.events
        measured = ClassicalSegmenter().measure(pr.camera_image())
        assert iou(measured, pr.printed[-1].region) > .9


@pytest.mark.parametrize("clear", [False, True])
def test_replacement_preserves_prefix_and_explicit_failure_semantics(clear):
    region = box(-5, -5, 5, 5)
    layers = [Layer(.25*(i+1), region) for i in range(8)]
    failure = random_failure(layers, np.random.default_rng(3), kind="shift", k0=0, severity=2)
    with MujocoPrinter(layers, failure, clear_failure_on_adapt=clear, render=False) as pr:
        first = pr.print_layer()
        pr.pause()
        with pytest.raises(RuntimeError):
            pr.print_layer()
        pr.replace_remaining(layers[1:])
        assert pr.printed[0] is first
        assert pr.failure_active == (not clear)
        assert pr.replacements and "G1" in pr.replacements[0]
        second = pr.print_layer()
        assert iou(second.region, region) > .98 if clear else iou(second.region, region) < .95


def test_rig_rejects_out_of_bed_plan_before_execution():
    with pytest.raises(ValueError, match="bed"):
        MujocoPrinter([Layer(.25, box(-100, -10, 100, 10))], profile="bedslinger", render=False)
