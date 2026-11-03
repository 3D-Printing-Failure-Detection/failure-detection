from copy import deepcopy

import numpy as np
import pytest

from recovery.eval.camera_trace import (
    camera_frame_digest, encode_camera_prefix, validate_camera_manifest,
    validate_camera_prefixes,
)


def row(method, pixels, adapt_at=None, total_frames=None):
    return dict(profile='cartesian', part='cup', kind='warp', seed=1003, method=method,
        frames=len(pixels) if total_frames is None else total_frames,
        actions=[] if adapt_at is None else [dict(layer=adapt_at, action='adapt')],
        camera_prefix=encode_camera_prefix([
            camera_frame_digest(np.full((8, 8, 3), value, dtype=np.uint8)) for value in pixels]))


def test_short_failed_prefix_cannot_hide_different_inputs_from_completed_peer():
    failed = row('neural_risk', [1, 2, 3, 99])
    completed = row('neural_risk_agent', [1, 2, 3, 4, 5, 6])
    for records in ([failed, completed], [completed, failed]):
        with pytest.raises(AssertionError, match='camera inputs differ before adaptation.*layer 3'):
            validate_camera_prefixes(records, required=True)


def test_matching_prefixes_allow_different_lengths_and_later_interventions():
    adapted = row('neural_risk', [1, 2, 3], adapt_at=2, total_frames=6)
    unadapted = row('neural_risk_agent', [1, 2, 3, 4, 5, 6])
    result = validate_camera_manifest({'camera_trace_version': 1}, [adapted, unadapted], required=True)
    assert result['checked_rows'] == 2 and result['unchecked_legacy_rows'] == 0
    assert result['paired_cases'] == 1 and result['compared_frames'] == 3
    truncated = deepcopy(unadapted)
    truncated['camera_prefix'] = adapted['camera_prefix']
    with pytest.raises(AssertionError, match='cover every frame'):
        validate_camera_prefixes([truncated], required=True)


@pytest.mark.parametrize('change, message', [
    ({'digests': 'invalid!'}, 'encoding'),
    ({'frames': 2}, 'fingerprint count'),
    ({'version': 2}, 'version'),
    ({'algorithm': 'unknown'}, 'algorithm'),
])
def test_malformed_camera_evidence_is_rejected(change, message):
    record = row('neural_risk', [1])
    record['camera_prefix'].update(change)
    with pytest.raises(AssertionError, match=message):
        validate_camera_prefixes([record], required=True)


def test_legacy_evidence_is_explicitly_unverified_and_can_be_prohibited():
    legacy = dict(method='old')
    assert validate_camera_manifest({}, [legacy])['unchecked_legacy_rows'] == 1
    with pytest.raises(AssertionError, match='declaration is missing'):
        validate_camera_manifest({}, [legacy], required=True)
    with pytest.raises(AssertionError, match='prefix is missing'):
        validate_camera_prefixes([legacy], required=True)
    with pytest.raises(AssertionError, match='declaration differs'):
        validate_camera_manifest({'camera_trace_version': 1}, [legacy])
    assert validate_camera_prefixes([row('ignore', [], total_frames=0)], required=True)['recorded_frames'] == 0


def test_fingerprint_reads_actual_pixels_without_modifying_them():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    before = image.copy()
    initial = camera_frame_digest(image)
    assert np.array_equal(image, before)
    image[0, 0, 0] = 1
    assert camera_frame_digest(image) != initial
    assert camera_frame_digest(image.reshape(8, 24)) != camera_frame_digest(image)


def test_capture_precedes_measurement_and_stops_after_intervention():
    from types import SimpleNamespace
    from shapely.geometry import box
    from recovery.eval.mujoco_eval import ObservedSegmenter
    from recovery.types import Layer

    actual = Layer(.25, box(0, 0, 1, 1))
    printer = SimpleNamespace(printed=[actual], replacements=[])

    class MutatingSegmenter:
        def measure(self, image, expected_mask):
            image[:] = 255
            return actual.region

    segmenter = ObservedSegmenter(printer, MutatingSegmenter())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = camera_frame_digest(image)
    segmenter.measure(image)
    assert segmenter.camera_prefix == [expected]
    printer.replacements.append('new commands')
    segmenter.measure(image)
    assert segmenter.camera_prefix == [expected]
