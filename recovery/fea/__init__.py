"""Voxel finite-element stress simulator for the part about to be printed.

    from recovery.fea import stress_test
    report = stress_test(layers, spec, layer_h)     # spec["loads"] defines the load cases
"""
from .cases import CaseResult, StressReport, resolve_solver, stress_test  # noqa: F401
from .material import MATERIALS, Material  # noqa: F401
