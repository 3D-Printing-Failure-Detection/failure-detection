import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch import nn

from recovery.models.boundary import BoundarySegmenter, normalized_gray
from tools.boundary_tiles import tile_inputs
from tools.train_boundary_tiles import TileNetwork
from tools.train_boundary_refinement import (
    export_refinement, exported_refined_tiles, frozen_tile_logits, verify_base_training,
    refinement_distance_loss,
)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_learned_refinement_matches_full_camera_across_native_block_seams(tmp_path, precision):
    torch.manual_seed(9)
    base_path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(base_path, precision)
    refinement = nn.Conv2d(1, 1, 3)
    with torch.no_grad():
        refinement.weight.copy_(torch.tensor([[[[.02, -.03, .01], [.04, .9, .06], [-.02, .03, .01]]]]))
        refinement.bias.fill_(.01)
    path = tmp_path/'refined.npz'
    export_refinement(base_path, refinement, path, precision)
    model = BoundarySegmenter(path)
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 255, (24, 32, 3), dtype='uint8') for _ in range(2)]
    expected = rng.integers(0, 2, (24, 32), dtype='uint8')
    frames = [*(normalized_gray(image) for image in images), expected.astype('float32')]
    model.prime_history(images[1])
    actual = model.signed_distance(images[0], expected)
    assert actual.dtype == np.dtype(precision)
    for x, y in [(0, 0), (16, 8), (3, 2)]:
        inputs = tile_inputs(frames, (x, y), 16)[None]
        logits = frozen_tile_logits(base_path, torch.from_numpy(inputs))
        with torch.no_grad():
            trained = torch.tanh(refinement(logits)).numpy()
        np.testing.assert_allclose(trained, exported_refined_tiles(model, inputs), atol=1e-6)
        np.testing.assert_allclose(trained[0, 0], actual[y*4+1:(y+16)*4-1, x*4+1:(x+16)*4-1], atol=1e-6)
    with pytest.raises(ValueError, match='spatial distance grid'):
        model.predict_patches(np.zeros((1, 243)))


def test_identity_refinement_preserves_existing_camera_and_weights(tmp_path):
    torch.manual_seed(5)
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    refinement = nn.Conv2d(1, 1, 3)
    with torch.no_grad():
        refinement.weight.zero_()
        refinement.weight[0, 0, 1, 1] = 1
        refinement.bias.zero_()
    updated = tmp_path/'identity.npz'
    export_refinement(path, refinement, updated, 'float32')
    with np.load(path) as old, np.load(updated) as new:
        assert all(np.array_equal(old[key], new[key]) for key in old.files)
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, (32, 32, 3), dtype='uint8')
    expected = rng.integers(0, 2, (32, 32), dtype='uint8')
    np.testing.assert_array_equal(BoundarySegmenter(path).signed_distance(image, expected),
                                  BoundarySegmenter(updated).signed_distance(image, expected))


def test_refinement_rejects_partial_and_invalid_weights(tmp_path):
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    for kernel, bias in [(np.eye(3), None), (np.eye(2), np.array(0.)),
                         (np.full((3, 3), np.nan), np.array(0.)),
                         (np.eye(3), np.array(np.inf)), (np.eye(3), np.zeros(2))]:
        changed = dict(arrays, native_refinement_kernel=kernel)
        if bias is not None:
            changed['native_refinement_bias'] = bias
        np.savez_compressed(tmp_path/'bad.npz', **changed)
        with pytest.raises(ValueError, match='native refinement weights'):
            BoundarySegmenter(tmp_path/'bad.npz')


def test_refinement_cannot_inherit_weights_trained_on_its_validation_split(tmp_path):
    checkpoint = tmp_path/'base.npz'
    checkpoint.write_bytes(b'identity only; not loaded for this provenance check')
    groups = np.array(['train-a', 'val-b'])
    validation = np.array([False, True])
    manifest = dict(status='complete', checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    data_sha256='same-data', training_groups=['train-a'], validation_groups=['val-b'])
    path = tmp_path/'training.json'
    path.write_text(json.dumps(manifest))
    assert verify_base_training(checkpoint, 'same-data', groups, validation)['path'] == str(path)
    for field, value in [('training_groups', ['train-a', 'val-b']),
                         ('data_sha256', 'other-data'), ('checkpoint_sha256', 'other-weights')]:
        path.write_text(json.dumps(dict(manifest, **{field:value})))
        with pytest.raises(ValueError, match='training provenance'):
            verify_base_training(checkpoint, 'same-data', groups, validation)


def test_gap_sensitive_cost_preserves_control_and_penalizes_invented_material():
    prediction = torch.tensor([.1, -.1], requires_grad=True)
    target = torch.tensor([-.1, .1])
    control = torch.nn.functional.smooth_l1_loss(prediction, target, beta=.05, reduction='none')
    assert torch.equal(refinement_distance_loss(prediction, target), control)
    loss = refinement_distance_loss(prediction, target, 2.)
    assert torch.equal(loss, control*torch.tensor([2., 1.]))
    loss.sum().backward()
    assert torch.equal(prediction.grad, torch.tensor([2., -1.]))
    assert float(refinement_distance_loss(target, target, 2.).sum()) == 0
    for weight in [.5, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='overprediction weight'):
            refinement_distance_loss(prediction, target, weight)
