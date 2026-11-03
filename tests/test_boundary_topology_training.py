import numpy as np
import pytest

torch = pytest.importorskip('torch')
from recovery.models.boundary import BoundarySegmenter, normalized_gray
from tools.boundary_tiles import tile_inputs
from tools.train_boundary_tiles import TileNetwork, exported_tiles, soft_skeleton, connectivity_losses


def test_dense_convolution_export_matches_actual_opencv_camera_inference(tmp_path):
    torch.manual_seed(7)
    network = TileNetwork()
    path = tmp_path/'dense.npz'
    network.export(path)
    model = BoundarySegmenter(path)
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 255, (24, 32, 3), dtype='uint8') for _ in range(2)]
    frames = [normalized_gray(image) for image in images]
    model.prime_history(images[1])
    actual = model.signed_distance(images[0])
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
