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
    refinement_distance_loss, ResidualRefinement, guided_refinement_inputs,
)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('nonlinear', [False, True])
@pytest.mark.parametrize('guided', [False, True])
def test_learned_refinement_matches_full_camera_across_native_block_seams(tmp_path, precision, nonlinear, guided):
    torch.manual_seed(9)
    base_path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(base_path, precision)
    if nonlinear:
        refinement = ResidualRefinement(8, 4 if guided else 1)
        with torch.no_grad():
            refinement.output.weight.uniform_(-.2, .2)
            refinement.output.bias.fill_(.01)
    else:
        refinement = nn.Conv2d(4 if guided else 1, 1, 3)
        with torch.no_grad():
            refinement.weight[0, 0].copy_(torch.tensor([[.02, -.03, .01], [.04, .9, .06], [-.02, .03, .01]]))
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
        if guided:
            logits = guided_refinement_inputs(logits, torch.from_numpy(inputs))
        with torch.no_grad():
            trained = torch.tanh(refinement(logits)).numpy()
        np.testing.assert_allclose(trained, exported_refined_tiles(model, inputs), atol=1e-6)
        np.testing.assert_allclose(trained[0, 0], actual[y*4+1:(y+16)*4-1, x*4+1:(x+16)*4-1], atol=1e-6)
    with pytest.raises(ValueError, match='spatial distance grid'):
        model.predict_patches(np.zeros((1, 243)))


@pytest.mark.parametrize('nonlinear', [False, True])
@pytest.mark.parametrize('guided', [False, True])
def test_identity_refinement_preserves_existing_camera_and_weights(tmp_path, nonlinear, guided):
    torch.manual_seed(5)
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    if nonlinear:
        refinement = ResidualRefinement(8, 4 if guided else 1)
    else:
        refinement = nn.Conv2d(4 if guided else 1, 1, 3)
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


def test_native_guidance_uses_ordered_parent_pixel_observations(tmp_path):
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    refinement = nn.Conv2d(4, 1, 3)
    with torch.no_grad():
        refinement.weight.zero_()
        refinement.bias.zero_()
        refinement.weight[0, 1:, 1, 1] = torch.tensor([1., 2., 4.])
    updated = tmp_path/'guided.npz'
    export_refinement(path, refinement, updated, 'float32')
    model = BoundarySegmenter(updated)
    frames = np.arange(3*6*7, dtype='float32').reshape(3, 6, 7)
    grid = model.refine_native_logits(np.zeros((24, 28), dtype='float32'), frames)
    for y, x in [(0, 0), (3, 3), (4, 4), (9, 13), (23, 27)]:
        assert grid[y, x] == sum(weight * frames[c, y//4, x//4] for c, weight in enumerate([1, 2, 4]))
    tiles = tile_inputs(frames, (1, 1), 4)[None]
    inputs = guided_refinement_inputs(torch.zeros((1, 1, 16, 16)), torch.from_numpy(tiles))
    for y, x in [(0, 0), (3, 7), (12, 15)]:
        np.testing.assert_array_equal(inputs[0, 1:, y, x].numpy(), frames[:, 1+y//4, 1+x//4])
    for bad in [None, frames[:2], np.zeros((3, 5, 7)), np.full_like(frames, np.nan)]:
        with pytest.raises(ValueError, match='native guidance'):
            model.refine_native_logits(np.zeros((24, 28)), bad)


def test_native_guidance_cannot_silently_load_incompatible_model_inputs(tmp_path):
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    export_refinement(path, nn.Conv2d(4, 1, 3), tmp_path/'guided.npz', 'float32')
    with np.load(tmp_path/'guided.npz') as data:
        arrays = {key: data[key] for key in data.files}
    cases = [dict(arrays, native_guidance=np.array(1)),
             dict(arrays, native_guidance=np.array([True])),
             {key: value for key, value in arrays.items() if key != 'native_guidance'},
             {key: value for key, value in arrays.items() if not key.startswith('native_refinement')},
             dict(arrays, native_refinement_kernel=np.zeros((3, 3, 3)))]
    for changed in cases:
        np.savez_compressed(tmp_path/'bad.npz', **changed)
        with pytest.raises(ValueError, match='native'):
            BoundarySegmenter(tmp_path/'bad.npz')


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


def test_residual_refinement_rejects_partial_conflicting_and_invalid_weights(tmp_path):
    path = tmp_path/'base.npz'
    TileNetwork(include_expected_mask=True).export(path)
    export_refinement(path, ResidualRefinement(8), tmp_path/'residual.npz', 'float32')
    with np.load(tmp_path/'residual.npz') as data:
        arrays = {key: data[key] for key in data.files}
    cases = [dict(arrays, native_residual_kernels=np.zeros((8, 2, 2))),
             dict(arrays, native_residual_kernels=np.zeros((3, 3))),
             dict(arrays, native_residual_kernels=np.zeros((0, 3, 3))),
             dict(arrays, native_residual_bias=np.zeros(7)),
             dict(arrays, native_residual_output_weights=np.full(8, np.nan)),
             dict(arrays, native_residual_output_bias=np.array(np.inf)),
             dict(arrays, native_residual_output_bias=np.zeros(1)),
             dict(arrays, native_refinement_kernel=np.eye(3), native_refinement_bias=np.array(0.))]
    for key in ['native_residual_kernels', 'native_residual_bias',
                'native_residual_output_weights', 'native_residual_output_bias']:
        cases.append({name: value for name, value in arrays.items() if name != key})
    for changed in cases:
        np.savez_compressed(tmp_path/'bad.npz', **changed)
        with pytest.raises(ValueError, match='native.*refinement'):
            BoundarySegmenter(tmp_path/'bad.npz')
    inputs = torch.zeros((1, 3, 24, 24))
    with pytest.raises(ValueError, match='unrefined native camera'):
        frozen_tile_logits(tmp_path/'residual.npz', inputs)


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


def test_refinement_binds_augmented_base_to_exact_extra_dataset_and_sampling(tmp_path):
    checkpoint = tmp_path/'base.npz'
    checkpoint.write_bytes(b'provenance only')
    groups = np.array(['train', 'validation', 'extra'])
    split = np.array([False, True, False])
    extra = dict(sha256='extra-data', parent_data_sha256='parent', rows=1, groups=1,
                 extra_sampling_mass=.5, samples_per_epoch=1)
    manifest = dict(status='complete', checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    data_sha256='parent', training_groups=['extra', 'train'], validation_groups=['validation'],
                    extra_training=extra)
    (tmp_path/'training.json').write_text(json.dumps(manifest))
    verify_base_training(checkpoint, 'parent', groups, split, extra)
    for changed in (None, dict(extra, sha256='other-data'), dict(extra, extra_sampling_mass=.2)):
        with pytest.raises(ValueError, match='extra training provenance'):
            verify_base_training(checkpoint, 'parent', groups, split, changed)


def test_native_interior_loss_weights_exclude_crop_edges_and_validation():
    from tools.boundary_critical_material import critical_loss_factors
    target = np.zeros((3, 1, 4, 4), dtype='float32')
    masks = np.zeros_like(target, dtype=bool)
    target[:, :, 0, :] = .1; masks[:, :, 0, :] = True
    target[:, :, 1, 1] = .1; masks[:, :, 1, 1] = True
    validation = np.array([False, True, False])
    weights = critical_loss_factors(target[:, :, 1:-1, 1:-1], masks[:, :, 1:-1, 1:-1],
                                    validation, np.array([.5, .5]), .5)
    assert weights['critical_training_pixels'] == 2
    masks[validation] = False
    assert weights == critical_loss_factors(target[:, :, 1:-1, 1:-1], masks[:, :, 1:-1, 1:-1],
                                             validation, np.array([.5, .5]), .5)
