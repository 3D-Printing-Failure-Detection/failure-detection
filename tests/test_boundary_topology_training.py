import numpy as np
import pytest

torch = pytest.importorskip('torch')
from recovery.models.boundary import BoundarySegmenter, normalized_gray
from tools.boundary_tiles import tile_inputs
from tools.train_boundary_tiles import (TileNetwork, exported_tiles, soft_skeleton,
                                       connectivity_losses, training_sampling_probabilities,
                                       distance_gradient_loss, distance_error_loss)


@pytest.mark.parametrize('include_expected_mask,include_previous_expected_mask',
                         [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('inference_dtype', ['float32', 'float64'])
def test_dense_convolution_export_matches_actual_opencv_camera_inference(tmp_path, include_expected_mask,
        include_previous_expected_mask, inference_dtype):
    torch.manual_seed(7)
    network = TileNetwork(include_expected_mask=include_expected_mask,
                          include_previous_expected_mask=include_previous_expected_mask)
    path = tmp_path/'dense.npz'
    network.export(path, inference_dtype)
    model = BoundarySegmenter(path)
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 255, (24, 32, 3), dtype='uint8') for _ in range(2)]
    frames = [normalized_gray(image) for image in images]
    expected_mask = rng.integers(0, 2, (24, 32), dtype='uint8')
    previous_mask = rng.integers(0, 2, (24, 32), dtype='uint8')
    if include_expected_mask:
        frames.append(expected_mask.astype('float32'))
    if include_previous_expected_mask:
        frames.append(previous_mask.astype('float32'))
    model.prime_history(images[1], previous_mask)
    actual = model.signed_distance(images[0], expected_mask)
    assert actual.dtype == np.dtype(inference_dtype)
    if include_expected_mask:
        with pytest.raises(ValueError, match='requires the commanded'):
            model.signed_distance(images[0])
        with pytest.raises(ValueError, match='invalid commanded'):
            model.signed_distance(images[0], np.zeros((10, 10)))
    for x, y in [(0, 0), (16, 8), (3, 2)]:
        inputs = tile_inputs(frames, (x, y), 16)[None]
        with torch.no_grad():
            trained = network(torch.from_numpy(inputs)).numpy()
        np.testing.assert_allclose(trained, exported_tiles(model, inputs), atol=1e-6)
        np.testing.assert_allclose(trained[0, 0], actual[y*4:(y+16)*4, x*4:(x+16)*4], atol=1e-6)


def test_cldice_penalizes_a_broken_wall_and_a_hallucinated_bridge():
    truth = torch.zeros(1, 1, 32, 32)
    truth[:, :, 15, 3:29] = 1
    broken = truth.clone()
    broken[:, :, 15, 15:17] = 0
    for target, prediction in [(truth, broken), (broken, truth)]:
        skeleton = soft_skeleton(target, 20)
        _, perfect = connectivity_losses(target, target, skeleton, 20)
        _, bad = connectivity_losses(prediction, target, skeleton, 20)
        assert float(perfect) < 1e-6
        assert float(bad) > .03
    logits = torch.zeros(3, 1, 32, 32, requires_grad=True)
    labels = torch.cat([truth, torch.zeros_like(truth), torch.ones_like(truth)])
    dice, cldice = connectivity_losses(torch.sigmoid(logits), labels, soft_skeleton(labels, 20), 20)
    (dice+cldice).mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 15, 16] < 0
    assert cldice[1:].tolist() == [0., 0.]


def test_critical_sampling_uses_only_training_rows_and_bound_data():
    groups = np.array(['train-a', 'train-b', 'val-a', 'train-c', 'val-b'])
    validation = np.array([False, False, True, False, True])
    audit = dict(status='complete', data_sha256='bound-data', verified_target_rows=5, total_rows=5,
        rows=[dict(dataset_row=i, group=groups[i], validation=bool(validation[i]),
                   critical_positive_points=10) for i in (1, 2)])
    probability, count = training_sampling_probabilities(audit, groups, validation, 'bound-data', .5)
    np.testing.assert_allclose(probability, [.25, .5, .25])
    assert count == 1  # The critical validation tile cannot affect training mass.
    with pytest.raises(ValueError, match='dataset'):
        training_sampling_probabilities(audit, groups, validation, 'different-data', .5)
    audit['rows'][0]['validation'] = True
    with pytest.raises(ValueError, match='row differs'):
        training_sampling_probabilities(audit, groups, validation, 'bound-data', .5)


def test_gradient_supervision_penalizes_invented_holes_but_preserves_real_edges():
    target = torch.ones(1, 1, 8, 8)*.2
    # Equal pointwise absolute errors: a uniform bias and a spurious hole have
    # different spatial geometry. The original distance loss still handles bias.
    bias = target-.3
    hole = target+.3
    hole[:, :, 3:5, 3:5] = target[:, :, 3:5, 3:5]-.3
    assert torch.allclose(torch.abs(bias-target), torch.abs(hole-target))
    assert float(distance_gradient_loss(bias, target, .0625)) == 0
    assert float(distance_gradient_loss(hole, target, .0625)) > 0
    hole.requires_grad_()
    distance_gradient_loss(hole, target, .0625).backward()
    assert hole.grad[0, 0, 3, 3] < 0  # Restore material at the invented hole.
    assert hole.grad[0, 0, 2, 3] > 0  # Reduce the artificial step around it.
    assert float(distance_gradient_loss(hole.detach(), hole.detach(), .0625)) == 0
    # A real sharp edge has zero loss when matched, unlike generic smoothing.
    edge = target.clone()
    edge[:, :, :, :4] = -.2
    assert float(distance_gradient_loss(edge, edge, .0625)) == 0
    with pytest.raises(ValueError, match='spacing'):
        distance_gradient_loss(target, target, 0.)


def test_asymmetric_distance_cost_preserves_control_and_still_penalizes_filled_gaps():
    prediction = torch.tensor([-.1, .1], requires_grad=True)
    target = torch.tensor([.1, -.1])  # Missed material, then invented material.
    control = torch.nn.functional.smooth_l1_loss(prediction, target, beta=.05, reduction='none')
    assert torch.equal(distance_error_loss(prediction, target), control)
    weighted = distance_error_loss(prediction, target, 3.)
    assert torch.equal(weighted, control*torch.tensor([3., 1.]))
    weighted.sum().backward()
    assert torch.equal(prediction.grad, torch.tensor([-3., 1.]))
    assert float(distance_error_loss(target, target, 3.).sum()) == 0
    for invalid in [.5, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='underprediction weight'):
            distance_error_loss(prediction, target, invalid)
