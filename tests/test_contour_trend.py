from collections import deque

import numpy as np
from shapely.geometry import box

from recovery.loop.contour import ContourObservationState, ContourPolicy
from recovery.loop.camera_fallback import FallbackNeuralPolicy
from recovery.loop.neural import NeuralPolicy
from recovery.types import Layer


def history(increment, *, enabled, uncertainty=.125):
    state = ContourObservationState(shrink_trend=enabled)
    state.uncertainty_mm = uncertainty
    state.residuals = deque([lambda points, role, n=n:
                            np.broadcast_to(n*np.asarray(increment), points.shape).copy()
                            for n in range(4)], maxlen=4)
    return state


def test_subresolution_trend_is_removed_without_erasing_current_deformation():
    points = np.zeros((3, 2))
    state = history([.03, .04], enabled=True)
    expected = np.broadcast_to([.09, .12], points.shape)
    np.testing.assert_allclose(state.displacement(points, 1), expected)
    np.testing.assert_allclose(state.displacement(points, 20), expected)
    assert state.trend_shrinkage_queries == 2
    default = history([.03, .04], enabled=False)
    np.testing.assert_allclose(default.displacement(points, 20), 2*expected)
    assert default.trend_shrinkage_queries == 0


def test_large_trend_keeps_direction_and_existing_horizon_limit():
    points = np.zeros((2, 2))
    state = history([.3, .4], enabled=True)
    # Norm .5 minus existing uncertainty .125 leaves .375 per layer.
    expected = np.broadcast_to(3*np.array([.3, .4]) + 3*np.array([.225, .3]), points.shape)
    np.testing.assert_allclose(state.displacement(points, 3), expected)
    np.testing.assert_allclose(state.displacement(points, 30), expected)
    np.testing.assert_allclose(state.displacement(points, 0), np.broadcast_to([.9, 1.2], points.shape))


def test_shrinkage_tracks_observed_uncertainty_and_handles_missing_history():
    points = np.zeros((2, 2))
    state = history([.3, .4], enabled=True, uncertainty=.5)
    np.testing.assert_allclose(state.displacement(points, 20), np.broadcast_to([.9, 1.2], points.shape))
    empty = ContourObservationState(shrink_trend=True)
    np.testing.assert_array_equal(empty.displacement(points), points)
    empty.residuals.append(lambda points, role: np.full_like(points, .2))
    np.testing.assert_array_equal(empty.displacement(points), np.full_like(points, .2))
    assert empty.trend_shrinkage_queries == 0


def test_both_camera_forecasts_receive_option_and_measurements_are_unchanged():
    from types import SimpleNamespace
    nominal = [Layer(z, box(-5, -5, 5, 5)) for z in [.25, .5, .75, 1.]]
    policy = FallbackNeuralPolicy({}, nominal, .25, checkpoint='unused-for-containers',
                                 forecast_scope='fits', shrink_trend=True,
                                 dual_segmenter=SimpleNamespace())
    assert policy.state.shrink_trend and policy.alternative.state.shrink_trend
    control = ContourPolicy({}, nominal, .25)
    for command in nominal:
        measured = Layer(command.z, box(-5+command.z*.03, -5, 5+command.z*.03, 5))
        assert policy.state.observe(command, measured).region.wkb == control.state.observe(command, measured).region.wkb
    before = [layer.region.wkb for layer in policy.state.observations]
    policy.state.predict(nominal)
    assert [layer.region.wkb for layer in policy.state.observations] == before


def test_fit_neural_model_is_not_changed_by_analytical_trend_option(monkeypatch):
    import recovery.loop.neural as module
    states = []

    def make_state(*args, **kwargs):
        state = ContourObservationState()
        states.append(state)
        return state

    monkeypatch.setattr(module, 'NeuralObservationState', make_state)
    nominal = [Layer(.25, box(-5, -5, 5, 5))]
    spec = {'holes': [{'center': [0, 0], 'diameter': [1, 3], 'tol_center': 1}]}
    policy = NeuralPolicy(spec, nominal, .25, checkpoint='trained', forecast_scope='fits', shrink_trend=True)
    assert policy.forecast_source == 'trained' and policy.state is states[0]
    assert not policy.state.shrink_trend
