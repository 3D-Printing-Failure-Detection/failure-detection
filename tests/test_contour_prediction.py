import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, box

from recovery.geometry.ops import hole_polys, iou, warp
from recovery.geometry.spec import check
from recovery.loop.contour import ContourObservationState, ContourPolicy
from recovery.models.classical import Detection, boundary_deviation
from recovery.types import Layer


def ring():
    return Point(0, 0).buffer(15).difference(Point(0, 0).buffer(13))


def test_contour_prediction_and_compensation_preserve_translation():
    nominal = Layer(.25, ring())
    state = ContourObservationState()
    for _ in range(4):
        state.observe(nominal, Layer(.25, translate(nominal.region, 1.5, -.5)))
    predicted = state.predict([nominal])[0]
    assert np.allclose(predicted.region.centroid.coords[0], [1.5, -.5], atol=.05)
    command = state.compensate([nominal])[0]
    assert iou(state.predict([command])[0].region, nominal.region) > .98
    state.observe(command, Layer(.25, translate(command.region, 1.5, -.5)))
    assert iou(state.predict([command])[0].region, nominal.region) > .97


def test_smooth_wall_motion_does_not_match_across_the_wall():
    nominal = Layer(.25, ring())

    def motion(points):
        weight = np.exp(-((points - [15., 0])**2).sum(axis=1) / 72.)
        return np.column_stack((3. * weight, np.zeros_like(weight)))

    observed = Layer(.25, warp(nominal.region, motion))
    state = ContourObservationState()
    for _ in range(4):
        state.observe(nominal, observed)
    predicted = state.predict([nominal])[0]
    assert len(hole_polys(predicted.region)) == len(hole_polys(observed.region)) == 1
    assert boundary_deviation(predicted.region, observed.region) < .3
    assert iou(predicted.region, observed.region) > .93


def test_separate_boundaries_still_predict_real_thinning():
    nominal = [Layer(.25 * (k + 1), ring()) for k in range(8)]
    observed = Layer(.25, nominal[0].region.buffer(-.45))
    state = ContourObservationState()
    for _ in range(4):
        state.observe(nominal[0], observed)
    predicted = state.predict(nominal)
    verdict = check(predicted, nominal, {'min_wall_mm': 1.4}, .25)
    assert not verdict.ok and verdict.metrics['thin_layers'] > 3
    assert iou(predicted[0].region, observed.region) > .95


def test_observed_open_wall_is_never_repaired_by_the_forecast():
    nominal = [Layer(.25 * (k + 1), ring()) for k in range(5)]
    broken = Layer(.25, ring().difference(box(0, -1, 20, 1)))
    spec = {'container': {'base_h': 0, 'capacity_mm3': [1, 1000]}}
    policy = ContourPolicy(spec, nominal, .25)
    measured, detection = policy.observe(nominal[0], broken, Detection('ok', .4, {}), 0)
    assert not hole_polys(measured.region)
    decision = policy.decide([measured], nominal, detection)
    assert decision.action == 'scrap'
    assert all(not c['ok'] for c in decision.considered)
