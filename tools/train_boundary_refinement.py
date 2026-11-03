"""Fit a native-grid spatial output layer while freezing the existing camera.

The learned 3x3 convolution acts on distance logits before tanh, not on a binary
mask. Valid convolutions exclude tile-edge padding from training and validation.
Full-image inference uses the same weights and reflect padding at image edges.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.models.boundary import BoundarySegmenter, PATCH_SIZE
from tools.train_boundary_tiles import TileNetwork, training_sampling_probabilities


def verify_base_training(checkpoint, data_sha256, groups, validation):
    path = checkpoint.parent/'training.json'
    training = json.loads(path.read_text())
    expected = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    data_sha256=data_sha256,
                    training_groups=sorted(set(groups[~validation].tolist())),
                    validation_groups=sorted(set(groups[validation].tolist())))
    if training.get('status') != 'complete' or any(training.get(key) != value for key, value in expected.items()):
        raise ValueError('base camera training provenance differs from the refinement split')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def frozen_tile_logits(checkpoint, inputs, batch_size=32):
    exported = BoundarySegmenter(checkpoint)
    if (exported.query_weights is not None or exported.native_refinement is not None
            or exported.input_scales != (1,) or exported.output_upsample != 4):
        raise ValueError('refinement requires an unrefined native camera with fine inputs')
    network = TileNetwork(exported.input_frames,
                          hidden=tuple(len(b) for _, b in exported.weights[:2]),
                          scale=exported.output_upsample,
                          include_expected_mask=exported.include_expected_mask)
    with torch.no_grad():
        for layer, (weight, bias) in zip(network.layers, exported.weights):
            layer.weight.copy_(torch.from_numpy(weight).reshape_as(layer.weight))
            layer.bias.copy_(torch.from_numpy(bias))
        outputs = []
        for start in range(0, len(inputs), batch_size):
            value = inputs[start:start+batch_size]
            for layer in network.layers[:-1]:
                value = torch.relu(layer(value))
            outputs.append(F.pixel_shuffle(network.layers[-1](value), network.scale))
    return torch.cat(outputs)


def export_refinement(base_checkpoint, layer, path, inference_dtype):
    with np.load(base_checkpoint, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays['native_refinement_kernel'] = layer.weight.detach().numpy()[0, 0].copy()
    arrays['native_refinement_bias'] = layer.bias.detach().numpy()[0].copy()
    if inference_dtype != 'float32':
        arrays['inference_dtype'] = np.array(inference_dtype)
    else:
        arrays.pop('inference_dtype', None)
    np.savez_compressed(path, **arrays)


def exported_refined_tiles(model, tiles):
    patches = np.lib.stride_tricks.sliding_window_view(
        tiles, (PATCH_SIZE, PATCH_SIZE), axis=(-2, -1))
    batch, channels, height, width = patches.shape[:4]
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, channels*PATCH_SIZE**2)
    scale = model.output_upsample
    logits = model.predict_patch_logits(patches).reshape(batch, height, width, scale, scale)
    logits = logits.transpose(0, 1, 3, 2, 4).reshape(batch, height*scale, width*scale)
    # The retained interior agrees with a valid Torch convolution, including at
    # subpixel block seams. No artificial crop-boundary padding is scored.
    return np.stack([np.tanh(model.refine_native_logits(grid))[1:-1, 1:-1]
                     for grid in logits])[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--sampling-audit', type=Path, required=True)
    parser.add_argument('--critical-tile-fraction', type=float, default=.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--inference-dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error('epochs and batch size must be positive')
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')) or args.out.exists():
        parser.error('output must be a new directory on the data volume')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    data_hash, base_hash = digest(args.data), digest(args.base_checkpoint)
    with np.load(args.data, allow_pickle=False) as data:
        x, y, validation, groups = [data[key] for key in ('x', 'y', 'validation', 'group')]
        assert int(data['input_frames']) == 2 and bool(data['include_expected_mask'])
        assert int(data['dense_tile_size']) == 16 and int(data['output_upsample']) == 4
        assert np.array_equal(data['input_scales'], [1])
    base = BoundarySegmenter(args.base_checkpoint)
    assert base.input_frames == 2 and base.include_expected_mask
    assert x.shape == (len(x), 3, 24, 24) and y.shape == (len(x), 1, 64, 64)
    assert validation.dtype == bool and validation.shape == groups.shape == (len(x),)
    assert validation.any() and (~validation).any()
    assert not set(groups[validation]) & set(groups[~validation])
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y)) <= 1
    assert np.isin(x[:, 2], [0, 1]).all()
    base_training = verify_base_training(args.base_checkpoint, data_hash, groups, validation)
    audit = json.loads(args.sampling_audit.read_text())
    probability, critical_count = training_sampling_probabilities(
        audit, groups, validation, data_hash, args.critical_tile_fraction)
    args.out.mkdir(parents=True)
    started = time.time()
    logits = frozen_tile_logits(args.base_checkpoint, torch.from_numpy(x.astype('float32')), args.batch_size)
    target = torch.from_numpy(y[:, :, 1:-1, 1:-1].astype('float32'))
    layer = nn.Conv2d(1, 1, 3)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, 0, 1, 1] = 1.
        layer.bias.zero_()
    initial_kernel = layer.weight.detach().numpy()[0, 0].tolist()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=.002, weight_decay=.0001)
    train_indices, val_indices = np.flatnonzero(~validation), np.flatnonzero(validation)

    def objective(prediction, indices):
        actual = target[indices]
        weight = .25 + .75*(torch.abs(actual) < .5)
        return (F.smooth_l1_loss(prediction, actual, beta=.05, reduction='none')*weight).mean()

    def evaluate(exported=None):
        score = error = edge_n = correct = near_n = maximum = 0.
        with torch.no_grad():
            for start in range(0, len(val_indices), args.batch_size):
                indices = val_indices[start:start+args.batch_size]
                predicted = torch.tanh(layer(logits[indices]))
                if exported is not None:
                    numpy_prediction = exported_refined_tiles(exported, x[indices].astype('float32'))
                    maximum = max(maximum, float(np.abs(numpy_prediction-predicted.numpy()).max()))
                    predicted = torch.from_numpy(numpy_prediction)
                actual = target[indices]
                score += float(objective(predicted, indices))*len(indices)
                edge, near = torch.abs(actual)<.5, torch.abs(actual)<.15
                error += float(torch.abs(predicted[edge]-actual[edge]).sum())
                edge_n += int(edge.sum())
                correct += int(((predicted[near]>0)==(actual[near]>0)).sum())
                near_n += int(near.sum())
        return dict(validation_selection_score=score/len(val_indices),
                    validation_boundary_mae_mm=error/max(1, edge_n),
                    near_boundary_sign_accuracy=correct/max(1, near_n),
                    export_max_abs_error=maximum)

    control = evaluate()
    export_refinement(args.base_checkpoint, layer, args.out/'identity.npz', args.inference_dtype)
    identity_verification = evaluate(BoundarySegmenter(args.out/'identity.npz'))
    assert identity_verification['export_max_abs_error'] <= 1e-5
    history, best, best_state = [], float('inf'), None
    checkpoint = args.out/'boundary.npz'
    for epoch in range(args.epochs):
        order = rng.choice(train_indices, size=len(train_indices), replace=True, p=probability)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start+args.batch_size]
            loss = objective(torch.tanh(layer(logits[indices])), indices)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        row = dict(epoch=epoch+1, train_loss=total/len(train_indices), **evaluate())
        history.append(row)
        if row['validation_selection_score'] < best:
            best = row['validation_selection_score']
            best_state = {name: value.clone() for name, value in layer.state_dict().items()}
            export_refinement(args.base_checkpoint, layer, checkpoint, args.inference_dtype)
        print(json.dumps(row), flush=True)
    layer.load_state_dict(best_state)
    verification = evaluate(BoundarySegmenter(checkpoint))
    with np.load(args.base_checkpoint, allow_pickle=False) as old, np.load(checkpoint, allow_pickle=False) as new:
        assert all(np.array_equal(old[key], new[key]) for key in old.files if key != 'inference_dtype')
    assert digest(args.base_checkpoint) == base_hash and digest(args.data) == data_hash
    export_passes = (verification['export_max_abs_error'] <= 1e-5
                     and abs(verification['validation_selection_score']-best) <= 1e-5)
    manifest = dict(status='complete' if export_passes else 'export_verification_failed',
        architecture='frozen commanded temporal camera -> native logits -> learned 3x3 convolution -> tanh',
        trained_parameters=10, base_weights_unchanged=True,
        base_checkpoint=str(args.base_checkpoint), base_checkpoint_sha256=base_hash,
        base_training_manifest=base_training,
        data_sha256=data_hash, checkpoint_sha256=digest(checkpoint), trainer_sha256=digest(Path(__file__)),
        training_groups=sorted(set(groups[~validation].tolist())), validation_groups=sorted(set(groups[validation].tolist())),
        train_rows=len(train_indices), validation_rows=len(val_indices),
        native_target_crop=1, initial_kernel=initial_kernel,
        kernel=layer.weight.detach().numpy()[0, 0].tolist(), bias=float(layer.bias.detach()[0]),
        sampling=dict(audit_sha256=digest(args.sampling_audit), critical_training_tiles=critical_count,
                      critical_probability_mass=args.critical_tile_fraction),
        control=control, identity_export_verification=identity_verification,
        inference_dtype=args.inference_dtype, torch=torch.__version__, device='cpu',
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        history=history, seconds=time.time()-started, **verification,
        limitation='Learned output refinement, not a fixed contour repair or calibrated confidence bound. Tile validation cannot establish gap preservation or successful recovery.')
    (args.out/('training.json' if export_passes else 'export-failure.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    if not export_passes:
        raise ValueError('exported native refinement differs from the trained model')
    print(json.dumps({key:manifest[key] for key in ['checkpoint_sha256', 'kernel', 'bias', *verification]}), flush=True)


if __name__ == '__main__':
    main()
