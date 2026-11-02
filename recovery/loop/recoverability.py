"""Analytical recovery tools inspired by HW4_Print_Recovery_Assistant.ipynb.

Source: https://colab.research.google.com/drive/1qsvnBkzzJ74yCOGFyT7BJHdNe8Nc2IIN
The correction limit is an explicit assumption, not a measured printer capability.
Cost inputs are hypothetical unless supplied from measurements. No probability is
inferred from a geometry margin.
"""
import math
from numbers import Integral, Real


def _finite_nonnegative(name, value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")


def correction_budget(correction_mm, max_step_mm, remaining_layers, layer_h):
    """Lower-bound layers to remove a scalar displacement at an assumed rate."""
    _finite_nonnegative("correction_mm", correction_mm)
    _finite_nonnegative("max_step_mm", max_step_mm, positive=True)
    _finite_nonnegative("layer_h", layer_h, positive=True)
    if isinstance(remaining_layers, bool) or not isinstance(remaining_layers, Integral) or remaining_layers < 0:
        raise ValueError("remaining_layers must be a nonnegative integer")
    ratio = correction_mm / max_step_mm
    nearest = round(ratio)
    # Tolerate only floating-point rounding at integer boundaries.
    if math.isclose(ratio, nearest, rel_tol=0, abs_tol=4 * math.ulp(ratio)):
        ratio = nearest
    required = math.ceil(ratio)
    return dict(required_layers=required, remaining_layers=int(remaining_layers),
                transition_height_mm=required * layer_h, spare_layers=int(remaining_layers) - required,
                nominal_reachable=required <= remaining_layers, assumed_max_step_mm=max_step_mm)


def intervention_tradeoff(continue_cost, recovery_cost, intervention_cost, restart_cost,
                          *, p_continue=None, p_recovery=None):
    """Compare recovery to continuing, in one consistent cost unit (e.g. hours).

As in the notebook, an unsuccessful attempt is followed by a hypothetical full
restart. Break-even requires p_recovery - p_continue to exceed the returned
threshold. With unknown probabilities, report the threshold without a decision.
"""
    for name, value in (("continue_cost", continue_cost), ("recovery_cost", recovery_cost),
                        ("intervention_cost", intervention_cost)):
        _finite_nonnegative(name, value)
    _finite_nonnegative("restart_cost", restart_cost, positive=True)
    added = recovery_cost + intervention_cost - continue_cost
    result = dict(required_success_gain=added / restart_cost, expected_savings=None,
                  recovery_economically_preferred=None)
    if (p_continue is None) != (p_recovery is None):
        raise ValueError("supply both success probabilities or neither")
    if p_continue is not None:
        for name, value in (("p_continue", p_continue), ("p_recovery", p_recovery)):
            _finite_nonnegative(name, value)
            if value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        saving = (p_recovery - p_continue) * restart_cost - added
        result.update(expected_savings=saving, recovery_economically_preferred=saving > 0)
    return result
