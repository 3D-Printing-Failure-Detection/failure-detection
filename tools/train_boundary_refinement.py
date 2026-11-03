"""Fit a native-grid spatial output layer while freezing the existing camera.

The learned spatial head acts on distance logits before tanh, not on a binary
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


class ResidualRefinement(nn.Module):
    """An identity-initialized nonlinear correction on the native logit grid."""
    def __init__(self, hidden_channels, input_channels=1):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError('residual refinement needs positive hidden channels')
        if input_channels not in (1, 4):
            raise ValueError('refinement inputs must be logits alone or logits plus three image channels')
        self.hidden = nn.Conv2d(input_channels, hidden_channels, 3)
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, logits):
        return logits[:, :1, 1:-1, 1:-1] + self.output(torch.relu(self.hidden(logits)))


def refinement_parameters(layer):
    if isinstance(layer, ResidualRefinement):
        kernels = layer.hidden.weight.detach().numpy()
        return dict(hidden_kernels=(kernels[:, 0] if layer.hidden.in_channels == 1 else kernels).tolist(),
                    hidden_bias=layer.hidden.bias.detach().numpy().tolist(),
                    output_weights=layer.output.weight.detach().numpy()[0, :, 0, 0].tolist(),
                    output_bias=float(layer.output.bias.detach()[0]))
    kernels = layer.weight.detach().numpy()[0]
    return dict(kernel=(kernels[0] if layer.in_channels == 1 else kernels).tolist(),
                bias=float(layer.bias.detach()[0]))


def refinement_distance_loss(prediction, target, overprediction_weight=1.):
    """Penalize excessive signed distance without claiming calibrated confidence."""
    if not np.isfinite(overprediction_weight) or overprediction_weight < 1:
        raise ValueError('overprediction weight must be finite and at least one')
    loss = F.smooth_l1_loss(prediction, target, beta=.05, reduction='none')
    if overprediction_weight == 1:
        return loss
    return torch.where(prediction > target, loss*overprediction_weight, loss)


def verify_base_training(checkpoint, data_sha256, groups, validation, extra_training=None):
    path = checkpoint.parent/'training.json'
    training = json.loads(path.read_text())
    expected = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    data_sha256=data_sha256,
                    training_groups=sorted(set(groups[~validation].tolist())),
                    validation_groups=sorted(set(groups[validation].tolist())))
    if training.get('status') != 'complete' or any(training.get(key) != value for key, value in expected.items()):
        raise ValueError('base camera training provenance differs from the refinement split')
    recorded_extra = training.get('extra_training')
    keys = ('sha256', 'parent_data_sha256', 'rows', 'groups', 'extra_sampling_mass', 'samples_per_epoch')
    if ((recorded_extra is None) != (extra_training is None)
            or extra_training is not None and any(recorded_extra.get(k) != extra_training[k] for k in keys)):
        raise ValueError('base camera extra training provenance differs from refinement data')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def frozen_tile_logits(checkpoint, inputs, batch_size=32):
    exported = BoundarySegmenter(checkpoint)
    if (exported.query_weights is not None or exported.has_native_refinement
            or exported.input_scales != (1,) or exported.output_upsample != 4):
        raise ValueError('refinement requires an unrefined native camera with fine inputs')
    network = TileNetwork(exported.input_frames,
                          hidden=tuple(len(b) for _, b in exported.weights[:2]),
                          scale=exported.output_upsample,
                          include_expected_mask=exported.include_expected_mask,
                          include_previous_expected_mask=exported.include_previous_expected_mask)
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
    if isinstance(layer, ResidualRefinement):
        channels = layer.hidden.in_channels
        kernels = layer.hidden.weight.detach().numpy()
        arrays['native_residual_kernels'] = (kernels[:, 0] if channels == 1 else kernels).copy()
        arrays['native_residual_bias'] = layer.hidden.bias.detach().numpy().copy()
        arrays['native_residual_output_weights'] = layer.output.weight.detach().numpy()[0, :, 0, 0].copy()
        arrays['native_residual_output_bias'] = layer.output.bias.detach().numpy()[0].copy()
    else:
        channels = layer.in_channels
        kernels = layer.weight.detach().numpy()[0]
        arrays['native_refinement_kernel'] = (kernels[0] if channels == 1 else kernels).copy()
        arrays['native_refinement_bias'] = layer.bias.detach().numpy()[0].copy()
    if channels not in (1, 4):
        raise ValueError('unsupported refinement input channels')
    if channels == 4:
        arrays['native_guidance'] = np.array(True)
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
    radius = PATCH_SIZE // 2
    return np.stack([np.tanh(model.refine_native_logits(grid,
        tile[:, radius:-radius, radius:-radius] if model.native_guidance else None))[1:-1, 1:-1]
        for grid, tile in zip(logits, tiles)])[:, None]


def guided_refinement_inputs(logits, tiles):
    """Use existing observed inputs at each native output's parent pixel."""
    radius = PATCH_SIZE // 2
    frames = tiles[:, :, radius:-radius, radius:-radius]
    if (logits.ndim != 4 or logits.shape[1] != 1 or frames.ndim != 4 or frames.shape[1] != 3
            or frames.shape[0] != logits.shape[0] or tuple(n*4 for n in frames.shape[2:]) != logits.shape[2:]):
        raise ValueError('guided refinement tile inputs do not match native logits')
    return torch.cat([logits, frames.repeat_interleave(4, 2).repeat_interleave(4, 3)], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--sampling-audit', type=Path, required=True)
    parser.add_argument('--critical-tile-fraction', type=float, default=.5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--hidden-channels', type=int, default=0,
                        help='positive enables a nonlinear residual head; zero retains the linear control')
    parser.add_argument('--condition-on-images', action='store_true',
                        help='give the spatial head the existing current/previous grayscale and commanded mask at each parent pixel')
    parser.add_argument('--overprediction-weight', type=float, default=1.,
                        help='training cost multiplier when predicted signed distance exceeds its target')
    parser.add_argument('--extra-training-data', type=Path,
                        help='verified training-only augmentation with unchanged original validation')
    parser.add_argument('--extra-training-fraction', type=float,
                        help='sampling mass for augmentation; retains the original epoch sample count')
    parser.add_argument('--critical-material-labels', type=Path,
                        help='verified thin-material labels for training loss weights only')
    parser.add_argument('--critical-pixel-mass', type=float,
                        help='expected loss mass for critical pixels on the valid native-grid interior')
    parser.add_argument('--inference-dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error('epochs and batch size must be positive')
    if args.hidden_channels < 0:
        parser.error('hidden channels must be nonnegative')
    if not np.isfinite(args.overprediction_weight) or args.overprediction_weight < 1:
        parser.error('overprediction weight must be finite and at least one')
    if (args.extra_training_data is None) != (args.extra_training_fraction is None):
        parser.error('extra training data and fraction must be supplied together')
    if (args.critical_material_labels is None) != (args.critical_pixel_mass is None):
        parser.error('critical material labels and pixel mass must be supplied together')
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
    assert not base.include_previous_expected_mask, 'refinement training does not yet load command-history sidecars'
    assert x.shape == (len(x), 3, 24, 24) and y.shape == (len(x), 1, 64, 64)
    assert validation.dtype == bool and validation.shape == groups.shape == (len(x),)
    assert validation.any() and (~validation).any()
    assert not set(groups[validation]) & set(groups[~validation])
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(y)) <= 1
    assert np.isin(x[:, 2], [0, 1]).all()
    audit = json.loads(args.sampling_audit.read_text())
    probability, critical_count = training_sampling_probabilities(
        audit, groups, validation, data_hash, args.critical_tile_fraction)
    samples_per_epoch = int((~validation).sum())
    extra_training = None
    if args.extra_training_data:
        from tools.boundary_training_augmentation import load_extra_training
        x, y, validation, groups, probability, extra_training = load_extra_training(
            args.extra_training_data, x, y, validation, groups, data_hash,
            probability, args.extra_training_fraction)
    base_training = verify_base_training(args.base_checkpoint, data_hash, groups, validation, extra_training)
    critical_masks, critical_training = None, None
    if args.critical_material_labels:
        from tools.boundary_critical_material import load_critical_material, critical_loss_factors
        masks = load_critical_material(args.critical_material_labels, groups, validation, y, data_hash,
            extra_data_sha256=extra_training['sha256'] if extra_training else None)
        # The head uses valid 3x3 convolution. Weight only the same interior
        # coordinates used by its training targets and unchanged validation.
        masks = masks[:, :, 1:-1, 1:-1]
        factors = critical_loss_factors(y[:, :, 1:-1, 1:-1], masks, validation,
                                        probability, args.critical_pixel_mass)
        critical_masks = torch.from_numpy(masks.copy())
        critical_training = dict(**factors, path=str(args.critical_material_labels.resolve()),
            sha256=digest(args.critical_material_labels), native_target_crop=1,
            selection='unchanged unweighted valid-interior validation objective', labels_are_inputs=False)
    args.out.mkdir(parents=True)
    started = time.time()
    logits = frozen_tile_logits(args.base_checkpoint, torch.from_numpy(x.astype('float32')), args.batch_size)
    refinement_inputs = (guided_refinement_inputs(logits, torch.from_numpy(x.astype('float32')))
                         if args.condition_on_images else logits)
    input_channels = refinement_inputs.shape[1]
    target = torch.from_numpy(y[:, :, 1:-1, 1:-1].astype('float32'))
    if args.hidden_channels:
        layer = ResidualRefinement(args.hidden_channels, input_channels)
        architecture = ('frozen commanded temporal camera -> native logits -> '
                        '3x3 convolution + ReLU + 1x1 convolution + residual -> tanh')
    else:
        layer = nn.Conv2d(input_channels, 1, 3)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[0, 0, 1, 1] = 1.
            layer.bias.zero_()
        architecture = 'frozen commanded temporal camera -> native logits -> learned 3x3 convolution -> tanh'
    if args.condition_on_images:
        architecture += '; spatial head also receives current/previous grayscale and commanded mask at parent-pixel coordinates'
    initial_parameters = refinement_parameters(layer)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=.002, weight_decay=.0001)
    train_indices, val_indices = np.flatnonzero(~validation), np.flatnonzero(validation)

    def objective(prediction, indices, *, training=False):
        actual = target[indices]
        weight = .25 + .75*(torch.abs(actual) < .5)
        if training and critical_masks is not None:
            weight = weight * torch.where(critical_masks[indices],
                critical_training['critical_factor'], critical_training['ordinary_factor'])
        return (refinement_distance_loss(prediction, actual, args.overprediction_weight)*weight).mean()

    def evaluate(exported=None):
        score = error = edge_n = correct = near_n = maximum = 0.
        with torch.no_grad():
            for start in range(0, len(val_indices), args.batch_size):
                indices = val_indices[start:start+args.batch_size]
                predicted = torch.tanh(layer(refinement_inputs[indices]))
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
        order = rng.choice(train_indices, size=samples_per_epoch, replace=True, p=probability)
        total = 0.
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start+args.batch_size]
            loss = objective(torch.tanh(layer(refinement_inputs[indices])), indices, training=True)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        row = dict(epoch=epoch+1, train_loss=total/len(order), **evaluate())
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
    if extra_training:
        assert digest(args.extra_training_data) == extra_training['sha256']
    if critical_training:
        assert digest(args.critical_material_labels) == critical_training['sha256']
    export_passes = (verification['export_max_abs_error'] <= 1e-5
                     and abs(verification['validation_selection_score']-best) <= 1e-5)
    manifest = dict(status='complete' if export_passes else 'export_verification_failed',
        architecture=architecture,
        trained_parameters=sum(parameter.numel() for parameter in layer.parameters()), base_weights_unchanged=True,
        base_checkpoint=str(args.base_checkpoint), base_checkpoint_sha256=base_hash,
        base_training_manifest=base_training,
        data_sha256=data_hash, checkpoint_sha256=digest(checkpoint), trainer_sha256=digest(Path(__file__)),
        training_groups=sorted(set(groups[~validation].tolist())), validation_groups=sorted(set(groups[validation].tolist())),
        train_rows=len(train_indices), validation_rows=len(val_indices),
        samples_per_epoch=samples_per_epoch, extra_training=extra_training,
        critical_material_training=critical_training,
        native_target_crop=1, initial_parameters=initial_parameters,
        distance_objective=dict(beta_mm=.05, overprediction_weight=args.overprediction_weight,
                                underprediction_weight=1., calibrated_probability=False),
        **refinement_parameters(layer),
        sampling=dict(audit_sha256=digest(args.sampling_audit), critical_training_tiles=critical_count,
                      critical_probability_mass=args.critical_tile_fraction),
        control=control, identity_export_verification=identity_verification,
        inference_dtype=args.inference_dtype, torch=torch.__version__, device='cpu',
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        history=history, seconds=time.time()-started, **verification,
        limitation='Learned output refinement, not a fixed contour repair or calibrated confidence bound. Tile validation cannot establish gap preservation or successful recovery.')
    if not args.hidden_channels:
        manifest['initial_kernel'] = initial_parameters['kernel']
    (args.out/('training.json' if export_passes else 'export-failure.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    if not export_passes:
        raise ValueError('exported native refinement differs from the trained model')
    print(json.dumps({key:manifest[key] for key in ['checkpoint_sha256', 'trained_parameters', *verification]}), flush=True)


if __name__ == '__main__':
    main()
