import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from recovery import camera
from recovery.models.boundary import BoundarySegmenter, normalized_gray, subpixel_offsets


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


def test_output_grid_metadata_must_match_checkpoint_heads(tmp_path):
    weights = [(np.zeros((1, 81), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32')),
               (np.zeros((1, 1), dtype='float32'), np.zeros(1, dtype='float32'))]
    with pytest.raises(ValueError, match='boundary output'):
        BoundarySegmenter(checkpoint(tmp_path/'mismatch.npz', weights, output_upsample=np.array(4)))
