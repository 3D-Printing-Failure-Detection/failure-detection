import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from recovery import camera
from recovery.models.boundary import BoundarySegmenter, normalized_gray, subpixel_offsets, camera_patches


def checkpoint(path, weights, **metadata):
    arrays = dict(patch_size=np.array(9))
    arrays.update(metadata)
    for i, (w, b) in enumerate(weights):
        arrays[f'w{i}'], arrays[f'b{i}'] = w, b
    np.savez(path, **arrays)
    return path


def test_full_frame_inference_matches_patch_network(tmp_path):
    rng = np.random.default_rng(71)
    weights = [(rng.normal(0, .05, (outputs, inputs)).astype('float32'),
                rng.normal(0, .03, outputs).astype('float32'))
               for inputs, outputs in ((81, 8), (8, 4), (4, 1))]
    model = BoundarySegmenter(checkpoint(tmp_path/'model.npz', weights))
    image = rng.integers(0, 255, (24, 32, 3), dtype='uint8')
    gray = normalized_gray(image)
    padded = np.pad(gray, 4, mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(padded, (9, 9)).reshape(-1, 81)
    for i, (w, b) in enumerate(weights):
        patches = patches @ w.T + b
        if i < 2:
            patches = np.maximum(patches, 0)
    expected = np.tanh(patches[:, 0]).reshape(gray.shape)
    assert np.allclose(model.signed_distance(image), expected, atol=1e-6)


def test_subpixel_contours_keep_camera_coordinate_origin(tmp_path):
    first = np.zeros((1, 81), dtype='float32')
    first[0, 40] = 1.
    weights = [(first, np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.array([-.5], dtype='float32'))]
    model = BoundarySegmenter(checkpoint(tmp_path/'model.npz', weights))
    yy, xx = np.mgrid[:camera.CAM.size, :camera.CAM.size]
    xy = camera.px_to_mm(np.column_stack([xx.ravel(), yy.ravel()]))
    centre = np.array([2.2, -1.3])
    radius = 10.
    distance = radius - np.linalg.norm(xy-centre, axis=1)
    gray = np.clip(127.5 + 50*distance, 0, 255).astype('uint8').reshape(xx.shape)
    region = model.measure(gray)
    assert np.linalg.norm(np.array(region.centroid.coords[0])-centre) < .03
    expected = Point(*centre).buffer(radius, quad_segs=128)
    assert region.symmetric_difference(expected).area / expected.area < .02


def test_invalid_boundary_weights_rejected(tmp_path):
    weights = [(np.ones((2, 80), dtype='float32'), np.zeros(2, dtype='float32'))] * 3
    with pytest.raises(ValueError, match='shape'):
        BoundarySegmenter(checkpoint(tmp_path/'bad.npz', weights))


def test_two_frame_inference_matches_observed_patches_at_both_scales(tmp_path):
    rng = np.random.default_rng(19)
    weights = [(rng.normal(0, .05, (outputs, inputs)).astype('float32'),
                rng.normal(0, .02, outputs).astype('float32'))
               for inputs, outputs in ((324, 8), (8, 4), (4, 16))]
    model = BoundarySegmenter(checkpoint(tmp_path/'temporal-camera.npz', weights,
        input_frames=np.array(2), input_scales=np.array([1, 4]), output_upsample=np.array(4)))
    previous = rng.integers(0, 255, (24, 32, 3), dtype='uint8')
    current = rng.integers(0, 255, (24, 32, 3), dtype='uint8')
    yy, xx = np.mgrid[:24, :32]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    patches = np.concatenate([camera_patches(normalized_gray(image), pixels, (1, 4))
                              for image in (current, previous)], axis=1)
    model.prime_history(previous)
    expected = model.predict_patches(patches).reshape(24, 32, 4, 4).transpose(0, 2, 1, 3).reshape(96, 128)
    assert np.allclose(model.signed_distance(current), expected, atol=1e-6)


def test_camera_history_uses_only_the_preceding_observation_and_resets(tmp_path):
    first = np.zeros((1, 162), dtype='float32')
    first[0, 81+40] = 1.  # Only the previous image's centre pixel matters.
    weights = [(first, np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.array([-.5], dtype='float32'))]
    model = BoundarySegmenter(checkpoint(tmp_path/'history.npz', weights, input_frames=np.array(2)))
    bright, dark = np.full((24, 32), 255, dtype='uint8'), np.zeros((24, 32), dtype='uint8')
    assert np.all(model.signed_distance(bright) > 0)  # Cold start duplicates current pixels.
    model.measure(bright)
    assert np.all(model.signed_distance(dark) > 0)  # Read-only inference must not advance history.
    assert np.all(model.signed_distance(dark) > 0)
    model.measure(dark)
    assert np.all(model.signed_distance(bright) < 0)
    assert model.queries == 2
    with pytest.raises(ValueError, match='different shape'):
        model.signed_distance(np.zeros((12, 12), dtype='uint8'))
    model.reset_history()
    assert np.all(model.signed_distance(bright) > 0)
    with pytest.raises(ValueError, match='input frames'):
        BoundarySegmenter(checkpoint(tmp_path/'bad-frames.npz', weights, input_frames=np.array(3)))


def test_native_outputs_represent_material_between_negative_pixel_centres(tmp_path):
    # A vertical strip occupies x=0.125 mm within each camera pixel. Sampling
    # only the original centres (x=0, 0.25, ...) would erase it entirely.
    offsets = subpixel_offsets(4)
    distance = .04 - np.abs(offsets[:, 0] / camera.CAM.px_per_mm - .125)
    weights = [(np.zeros((1, 81), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((16, 1), dtype='float32'), np.arctanh(distance).astype('float32'))]
    model = BoundarySegmenter(checkpoint(tmp_path/'native.npz', weights, output_upsample=np.array(4)))
    field = model.signed_distance(np.zeros((12, 16), dtype='uint8'))
    assert field.shape == (48, 64)
    assert np.all(field[:, ::4] < 0)
    assert np.all(field[:, 2::4] > 0)
    assert np.all(field[:, 1::4] < 0)
    assert np.all(field[:, 3::4] < 0)
    assert np.allclose(camera.px_to_mm([[256.5, 256.75]])[0], [.125, -.1875])
    with pytest.raises(ValueError, match='upsample factor'):
        BoundarySegmenter(tmp_path/'native.npz', upsample=1)


def test_previous_command_tracks_executed_masks_without_aliasing_or_advancing_on_reads(tmp_path):
    first = np.zeros((1, 324), dtype='float32')
    first[0, 3*81+40] = 1.  # Read only the previous command's centre pixel.
    weights = [(first, np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.array([-.5], dtype='float32'))]
    model = BoundarySegmenter(checkpoint(tmp_path/'commands.npz', weights,
        input_frames=np.array(2), include_expected_mask=np.array(True),
        include_previous_expected_mask=np.array(True)))
    image = np.zeros((24, 32), dtype='uint8')
    full, empty = np.ones_like(image), np.zeros_like(image)
    assert np.all(model.signed_distance(image, full) > 0)  # Cold start uses this command.
    model.measure(image, full)
    full[:] = 0  # Mutating the caller's buffer cannot rewrite recorded command history.
    assert np.all(model.signed_distance(image, empty) > 0)
    assert np.all(model.signed_distance(image, empty) > 0)
    model.measure(image, empty)
    full[:] = 1
    assert np.all(model.signed_distance(image, full) < 0)
    assert model.queries == 2
    with pytest.raises(ValueError, match='requires the previous commanded'):
        model.prime_history(image)
    assert np.all(model.signed_distance(image, full) < 0)
    with pytest.raises(ValueError, match='invalid previous commanded'):
        model.prime_history(image, np.zeros((3, 3)))
    model.prime_history(image, full)
    full[:] = 0
    assert np.all(model.signed_distance(image, empty) > 0)
    model.reset_history()
    assert np.all(model.signed_distance(image, empty) < 0)


@pytest.mark.parametrize('frames,current', [(1, True), (2, False)])
def test_previous_command_metadata_requires_matching_observed_context(tmp_path, frames, current):
    weights = [(np.zeros((1, 81*(frames+int(current)+1)), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.ones((1, 1), dtype='float32'), np.zeros(1, dtype='float32'))]
    with pytest.raises(ValueError, match='previous command requires'):
        BoundarySegmenter(checkpoint(tmp_path/'bad-command-history.npz', weights,
            input_frames=np.array(frames), include_expected_mask=np.array(current),
            include_previous_expected_mask=np.array(True)))


def test_output_grid_metadata_must_match_checkpoint_heads(tmp_path):
    weights = [(np.zeros((1, 81), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32'))]
    with pytest.raises(ValueError, match='boundary output'):
        BoundarySegmenter(checkpoint(tmp_path/'mismatch.npz', weights, output_upsample=np.array(4)))


def test_quantile_estimate_is_identified_and_invalid_metadata_rejected(tmp_path):
    weights = [(np.zeros((1, 81), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32'))]
    model = BoundarySegmenter(checkpoint(tmp_path/'upper.npz', weights, distance_quantile=np.array(.8)))
    assert model.distance_quantile == .8
    with pytest.raises(ValueError, match='distance quantile'):
        BoundarySegmenter(checkpoint(tmp_path/'invalid.npz', weights, distance_quantile=np.array(float('nan'))))


def test_shared_query_full_frame_matches_direct_coordinate_evaluation(tmp_path):
    rng = np.random.default_rng(27)
    weights = [(rng.normal(0, .1, (outputs, inputs)).astype('float32'),
                rng.normal(0, .1, outputs).astype('float32'))
               for inputs, outputs in ((81, 8), (8, 4), (4, 1))]
    queries = rng.normal(0, .1, (4, 2)).astype('float32')
    model = BoundarySegmenter(checkpoint(tmp_path/'query.npz', weights,
        output_upsample=np.array(4), query_weights=queries))
    image = rng.integers(0, 255, (12, 16, 3), dtype='uint8')
    gray = normalized_gray(image)
    patches = np.lib.stride_tricks.sliding_window_view(np.pad(gray, 4, mode='reflect'), (9, 9)).reshape(-1, 81)
    hidden = np.maximum(patches @ weights[0][0].T + weights[0][1], 0)
    hidden = hidden @ weights[1][0].T + weights[1][1]
    outputs = []
    for dx, dy in subpixel_offsets(4):
        local = np.maximum(hidden + dx*queries[:, 0] + dy*queries[:, 1], 0)
        outputs.append(np.tanh(local @ weights[2][0].T + weights[2][1])[:, 0])
    expected = np.stack(outputs, axis=-1)
    assert np.allclose(model.predict_patches(patches), expected, atol=1e-6)
    field = expected.reshape(12, 16, 4, 4).transpose(0, 2, 1, 3).reshape(48, 64)
    assert np.allclose(model.signed_distance(image), field, atol=1e-6)


def test_shared_query_decoder_can_represent_a_subpixel_wall(tmp_path):
    # d(x) = .04 - abs(.25*x - .125), using one shared two-ReLU
    # decoder, is positive between two negative camera-pixel centres.
    weights = [(np.zeros((1, 81), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((2, 1), dtype='float32'), np.array([0., -.5], dtype='float32')),
               (np.array([[.25, -.5]], dtype='float32'), np.array([-.085], dtype='float32'))]
    path = checkpoint(tmp_path/'wall.npz', weights, output_upsample=np.array(4),
                      query_weights=np.array([[1., 0.], [1., 0.]], dtype='float32'))
    field = BoundarySegmenter(path).signed_distance(np.zeros((8, 8), dtype='uint8'))
    assert np.all(field[:, ::4] < 0) and np.all(field[:, 2::4] > 0)
    assert np.all(field[:, 1::4] < 0) and np.all(field[:, 3::4] < 0)
    with pytest.raises(ValueError, match='query weights'):
        BoundarySegmenter(checkpoint(tmp_path/'bad-query.npz', weights, output_upsample=np.array(4),
                                    query_weights=np.zeros((2, 3), dtype='float32')))


def test_wider_context_export_matches_strided_patch_inputs_at_edges(tmp_path):
    rng = np.random.default_rng(31)
    weights = [(rng.normal(0, .05, (outputs, inputs)).astype('float32'),
                rng.normal(0, .02, outputs).astype('float32'))
               for inputs, outputs in ((162, 8), (8, 4), (4, 1))]
    model = BoundarySegmenter(checkpoint(tmp_path/'context.npz', weights, input_scales=np.array([1, 4])))
    image = rng.integers(0, 255, (23, 29), dtype='uint8')
    gray = normalized_gray(image)
    yy, xx = np.mgrid[:23, :29]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    patches = camera_patches(gray, pixels, (1, 4))
    reflected = np.pad(gray, 16, mode='reflect')
    expected = []
    for x, y in pixels:
        expected.append(np.concatenate([reflected[y+16-4*s:y+16+4*s+1:s,
                                                 x+16-4*s:x+16+4*s+1:s].ravel() for s in (1, 4)]))
    assert np.array_equal(patches, np.asarray(expected))
    assert np.allclose(model.signed_distance(image), model.predict_patches(patches).reshape(23, 29), atol=1e-6)
    with pytest.raises(ValueError, match='input scales'):
        BoundarySegmenter(checkpoint(tmp_path/'repeat-context.npz', weights, input_scales=np.array([1, 1])))
    with pytest.raises(ValueError, match='shape'):
        BoundarySegmenter(checkpoint(tmp_path/'missing-context.npz', weights))
