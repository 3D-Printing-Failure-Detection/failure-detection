"""Train the existing camera network on spatially adjacent distance labels.

Optional training-only soft-clDice follows Algorithms 1/2 of Shit et al.,
CVPR 2021, https://arxiv.org/abs/2003.07311. This is an implementation of the
published equations, not a runtime contour repair or a topology guarantee.
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
from recovery import camera


def distance_error_loss(prediction, target, underprediction_weight=1.):
    """Elementwise distance error with an explicit cost for underprediction.

    This is asymmetric regression, not a calibrated confidence bound. Missing
    real gaps must still be measured separately before accepting a checkpoint.
    """
    if not np.isfinite(underprediction_weight) or underprediction_weight < 1:
        raise ValueError('underprediction weight must be finite and at least one')
    loss = F.smooth_l1_loss(prediction, target, beta=.05, reduction='none')
    if underprediction_weight == 1:
        return loss
    return torch.where(prediction < target, loss*underprediction_weight, loss)


def distance_gradient_loss(prediction, target, spacing_mm):
    """Match adjacent native-grid distance changes, including pixel-block seams.

    This supervises the observed geometry's gradients, rather than smoothing
    every edge indiscriminately. Target gaps and thin walls retain their gradients.
    """
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError('distance grid spacing must be finite and positive')
    losses = []
    for axis in (-2, -1):
        predicted = torch.diff(prediction, dim=axis)/spacing_mm
        actual = torch.diff(target, dim=axis)/spacing_mm
        losses.append(F.smooth_l1_loss(predicted, actual, beta=.1))
    return .5*(losses[0]+losses[1])


def soft_skeleton(probability, iterations):
    # A cross-shaped erosion and square dilation approximate 2D thinning.
    def erode(value):
        vertical = -F.max_pool2d(-value, (3, 1), 1, (1, 0))
        horizontal = -F.max_pool2d(-value, (1, 3), 1, (0, 1))
        return torch.minimum(vertical, horizontal)

    def residue(value):
        return F.relu(value - F.max_pool2d(erode(value), 3, 1, 1))

    skeleton = residue(probability)
    for _ in range(iterations):
        probability = erode(probability)
        skeleton = skeleton + (1-skeleton) * residue(probability)
    return skeleton


def connectivity_losses(probability, truth, truth_skeleton, iterations):
    """Per-tile Dice and clDice losses; homogeneous tiles have no clDice term."""
    dims = (1, 2, 3)
    epsilon = 1e-6
    skeleton = soft_skeleton(probability, iterations)
    precision = ((skeleton*truth).sum(dims)+epsilon)/(skeleton.sum(dims)+epsilon)
    recall = ((truth_skeleton*probability).sum(dims)+epsilon)/(truth_skeleton.sum(dims)+epsilon)
    cldice = 1 - 2*precision*recall/(precision+recall).clamp_min(epsilon)
    mixed = (truth.sum(dims) > 0) & ((1-truth).sum(dims) > 0)
    cldice = torch.where(mixed, cldice, torch.zeros_like(cldice))
    dice = 1 - (2*(probability*truth).sum(dims)+epsilon)/(probability.sum(dims)+truth.sum(dims)+epsilon)
    return dice, cldice


class TileNetwork(nn.Module):
    def __init__(self, frames=2, hidden=(64, 32), scale=4, include_expected_mask=False,
                 include_previous_expected_mask=False):
        super().__init__()
        self.frames, self.scale = frames, scale
        self.include_expected_mask = include_expected_mask
        self.include_previous_expected_mask = include_previous_expected_mask
        if include_previous_expected_mask and (frames != 2 or not include_expected_mask):
            raise ValueError('previous command requires two observed frames and the current command')
        self.layers = nn.ModuleList([nn.Conv2d(frames+int(include_expected_mask)
                                             +int(include_previous_expected_mask), hidden[0], PATCH_SIZE),
                                    nn.Conv2d(hidden[0], hidden[1], 1),
                                    nn.Conv2d(hidden[1], scale**2, 1)])

    def forward(self, value):
        for layer in self.layers[:-1]:
            value = torch.relu(layer(value))
        return F.pixel_shuffle(torch.tanh(self.layers[-1](value)), self.scale)

    def export(self, path, inference_dtype='float32'):
        arrays = dict(patch_size=np.array(PATCH_SIZE), output_upsample=np.array(self.scale),
                      input_frames=np.array(self.frames), input_scales=np.array([1]))
        if self.include_expected_mask:
            arrays['include_expected_mask'] = np.array(True)
        if self.include_previous_expected_mask:
            arrays['include_previous_expected_mask'] = np.array(True)
        if inference_dtype != 'float32':
            arrays['inference_dtype'] = np.array(inference_dtype)
        for i, layer in enumerate(self.layers):
            arrays[f'w{i}'] = layer.weight.detach().numpy().reshape(layer.out_channels, -1)
            arrays[f'b{i}'] = layer.bias.detach().numpy()
        np.savez_compressed(path, **arrays)


def exported_tiles(model, tiles):
    """Independent NumPy patch inference, reassembled at native coordinates."""
    patches = np.lib.stride_tricks.sliding_window_view(tiles, (PATCH_SIZE, PATCH_SIZE), axis=(-2, -1))
    batch, channels, height, width = patches.shape[:4]
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, channels*PATCH_SIZE**2)
    scale = model.output_upsample
    return model.predict_patches(patches).reshape(batch, height, width, scale, scale).transpose(
        0, 1, 3, 2, 4).reshape(batch, 1, height*scale, width*scale)


def training_sampling_probabilities(audit, groups, validation, data_sha256, critical_fraction):
    """Balance audited thin-material tiles using training labels alone."""
    if not 0 < critical_fraction < 1:
        raise ValueError('critical fraction must be strictly between zero and one')
    if (audit['status'] != 'complete' or audit['data_sha256'] != data_sha256
            or audit['verified_target_rows'] != len(groups) or audit['total_rows'] != len(groups)):
        raise ValueError('sampling audit does not verify this dataset')
    critical = np.zeros(len(groups), dtype=bool)
    seen = set()
    for row in audit['rows']:
        index = row['dataset_row']
        if (index in seen or not 0 <= index < len(groups) or groups[index] != row['group']
                or bool(validation[index]) != row['validation'] or row['critical_positive_points'] < 0):
            raise ValueError('sampling audit row differs from the dataset')
        seen.add(index)
        critical[index] = row['critical_positive_points'] > 0
    selected = critical[~validation]
    if not selected.any() or selected.all():
        raise ValueError('training needs both critical and ordinary tiles')
    probability = np.where(selected, critical_fraction/selected.sum(),
                           (1-critical_fraction)/(~selected).sum())
    return probability, int(selected.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--topology-weight', type=float, default=0.)
    parser.add_argument('--gradient-weight', type=float, default=0.,
                        help='supervise adjacent signed-distance gradients on the native output grid')
    parser.add_argument('--underprediction-weight', type=float, default=1.,
                        help='training cost multiplier when predicted signed distance is below its target')
    parser.add_argument('--skeleton-iterations', type=int, default=20)
    parser.add_argument('--temperature-mm', type=float, default=.1)
    parser.add_argument('--sampling-audit', type=Path,
                        help='exact-target replay audit identifying thin-material training tiles')
    parser.add_argument('--critical-tile-fraction', type=float,
                        help='probability mass assigned to audited thin-material training tiles')
    parser.add_argument('--include-expected-mask', action='store_true',
                        help='use the saved commanded layer mask as an additional known input')
    parser.add_argument('--previous-command-masks', type=Path,
                        help='verified sidecar of preceding captured command masks; requires --include-expected-mask')
    parser.add_argument('--critical-material-labels', type=Path,
                        help='verified thin-material labels used only to weight training losses')
    parser.add_argument('--critical-pixel-mass', type=float,
                        help='fraction of expected training loss weight assigned to audited thin material')
    parser.add_argument('--inference-dtype', choices=['float32', 'float64'], default='float32',
                        help='exported accumulation precision; training and stored weights remain float32')
    args = parser.parse_args()
    if not np.isfinite(args.topology_weight) or args.topology_weight < 0:
        parser.error('topology weight must be finite and nonnegative')
    if not np.isfinite(args.gradient_weight) or args.gradient_weight < 0:
        parser.error('gradient weight must be finite and nonnegative')
    if not np.isfinite(args.underprediction_weight) or args.underprediction_weight < 1:
        parser.error('underprediction weight must be finite and at least one')
    if not np.isfinite(args.temperature_mm) or args.temperature_mm <= 0:
        parser.error('temperature must be finite and positive')
    if min(args.epochs, args.batch_size, args.skeleton_iterations) < 1:
        parser.error('epochs, batch size and skeleton iterations must be positive')
    if (args.sampling_audit is None) != (args.critical_tile_fraction is None):
        parser.error('sampling audit and critical tile fraction must be supplied together')
    if (args.critical_material_labels is None) != (args.critical_pixel_mass is None):
        parser.error('critical material labels and pixel mass must be supplied together')
    if not args.out.resolve().is_relative_to(Path('/mnt/data/ishaan')):
        parser.error('model artifacts must stay on the data volume')
    if args.out.exists():
        parser.error('refusing to overwrite a training run')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        x, y, split, groups = [data[k] for k in ('x', 'y', 'validation', 'group')]
        size, scale, frames = [int(data[k]) for k in ('dense_tile_size', 'output_upsample', 'input_frames')]
        has_expected_mask = bool(data['include_expected_mask']) if 'include_expected_mask' in data else False
        if not np.array_equal(data['input_scales'], [1]):
            raise ValueError('dense convolution requires fine camera inputs')
    if (size != 16 or scale != 4 or frames not in (1, 2)
            or x.shape != (len(x), frames+int(has_expected_mask), size+8, size+8)
            or y.shape != (len(x), 1, size*scale, size*scale)):
        raise ValueError('invalid dense camera dataset shapes')
    if (split.dtype != bool or split.shape != (len(x),) or groups.shape != (len(x),)
            or not split.any() or split.all() or set(groups[split]) & set(groups[~split])):
        raise ValueError('invalid trajectory-separated split')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y)) > 1:
        raise ValueError('invalid camera inputs or distance targets')
    if args.include_expected_mask and not has_expected_mask:
        parser.error('dataset has no commanded layer mask')
    if args.previous_command_masks and (not args.include_expected_mask or frames != 2):
        parser.error('previous command masks require two observed images and --include-expected-mask')
    if has_expected_mask and not np.isin(x[:, frames], [0, 1]).all():
        raise ValueError('commanded layer masks must be binary')
    if not args.include_expected_mask:
        x = x[:, :frames]
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    command_history = None
    if args.previous_command_masks:
        from tools.add_boundary_command_history import load_previous_commands
        previous = load_previous_commands(args.previous_command_masks, groups, split,
                                          (len(x), 1, size+8, size+8), data_sha256)
        original = x
        x = np.concatenate([x, previous.astype(x.dtype)], axis=1)
        assert np.array_equal(x[:, :-1], original)
        command_history = dict(path=str(args.previous_command_masks.resolve()),
            sha256=hashlib.sha256(args.previous_command_masks.read_bytes()).hexdigest(),
            parent_data_sha256=data_sha256, original_inputs_unchanged=True)
    sampling, sampling_metadata = None, None
    if args.sampling_audit:
        audit = json.loads(args.sampling_audit.read_text())
        sampling, critical_count = training_sampling_probabilities(
            audit, groups, split, data_sha256, args.critical_tile_fraction)
        sampling_metadata = dict(audit_sha256=hashlib.sha256(args.sampling_audit.read_bytes()).hexdigest(),
            critical_training_tiles=critical_count, critical_probability_mass=args.critical_tile_fraction,
            method='training-only weighted sampling with replacement; same samples per epoch',
            validation='unchanged full validation set and selection objective',
            inputs='unchanged observed images; thin labels affect training frequency only')
    args.out.mkdir(parents=True)
    checkpoint = args.out/'boundary.npz'
    started = time.time()
    inputs, target = torch.from_numpy(x.astype('float32')), torch.from_numpy(y.astype('float32'))
    truth = (target > 0).float()
    # Labels are fixed: compute their skeleton once, outside the gradient graph.
    skeleton = torch.zeros_like(truth)
    if args.topology_weight:
        for offset in range(0, len(x), args.batch_size):
            skeleton[offset:offset+args.batch_size] = soft_skeleton(truth[offset:offset+args.batch_size], args.skeleton_iterations)
    train_indices, val_indices = np.flatnonzero(~split), np.flatnonzero(split)
    critical_masks, critical_training = None, None
    if args.critical_material_labels:
        from tools.boundary_critical_material import load_critical_material, critical_loss_factors
        masks = load_critical_material(args.critical_material_labels, groups, split, y, data_sha256)
        factors = critical_loss_factors(y, masks, split, sampling, args.critical_pixel_mass)
        critical_masks = torch.from_numpy(masks)
        critical_training = dict(**factors, path=str(args.critical_material_labels.resolve()),
            sha256=hashlib.sha256(args.critical_material_labels.read_bytes()).hexdigest(),
            selection='unchanged unbalanced full-validation objective',
            labels_are_inputs=False)
    model = TileNetwork(frames, scale=scale, include_expected_mask=args.include_expected_mask,
                        include_previous_expected_mask=bool(args.previous_command_masks))
    initial_arrays = b''.join(t.detach().numpy().tobytes() for t in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)

    def objective(prediction, indices, *, training=False):
        actual = target[indices]
        weight = .25 + .75*(torch.abs(actual) < .5)
        if training and critical_masks is not None:
            weight = weight * torch.where(critical_masks[indices],
                critical_training['critical_factor'], critical_training['ordinary_factor'])
        loss = (distance_error_loss(prediction, actual, args.underprediction_weight)*weight).mean()
        if args.gradient_weight:
            loss = loss + args.gradient_weight * distance_gradient_loss(
                prediction, actual, 1/(camera.CAM.px_per_mm*scale))
        if args.topology_weight:
            dice, cldice = connectivity_losses(torch.sigmoid(prediction/args.temperature_mm),
                truth[indices], skeleton[indices], args.skeleton_iterations)
            loss = loss + args.topology_weight * (.5*dice + .5*cldice).mean()
        return loss

    def evaluate(exported=None):
        totals = dict(selection_score=0., boundary_error=0., boundary_n=0,
                      near_correct=0, near_n=0, export_max_abs_error=0.)
        with torch.no_grad():
            for offset in range(0, len(val_indices), args.batch_size):
                indices = val_indices[offset:offset+args.batch_size]
                predicted = model(inputs[indices])
                if exported is not None:
                    numpy_prediction = exported_tiles(exported, x[indices].astype('float32'))
                    totals['export_max_abs_error'] = max(totals['export_max_abs_error'],
                        float(np.abs(numpy_prediction-predicted.numpy()).max()))
                    predicted = torch.from_numpy(numpy_prediction)
                totals['selection_score'] += float(objective(predicted, indices))*len(indices)
                actual = target[indices]
                edge, near = torch.abs(actual)<.5, torch.abs(actual)<.15
                totals['boundary_error'] += float(torch.abs(predicted[edge]-actual[edge]).sum())
                totals['boundary_n'] += int(edge.sum())
                totals['near_correct'] += int(((predicted[near]>0)==(actual[near]>0)).sum())
                totals['near_n'] += int(near.sum())
        return dict(validation_selection_score=totals['selection_score']/len(val_indices),
                    validation_boundary_mae_mm=totals['boundary_error']/max(1,totals['boundary_n']),
                    near_boundary_sign_accuracy=totals['near_correct']/max(1,totals['near_n']),
                    export_max_abs_error=totals['export_max_abs_error'])

    history, best, best_state = [], float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        order = (rng.permutation(train_indices) if sampling is None else
                 rng.choice(train_indices, size=len(train_indices), replace=True, p=sampling))
        total = 0.
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset+args.batch_size]
            loss = objective(model(inputs[indices]), indices, training=True)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        model.eval()
        row = dict(epoch=epoch+1, train_loss=total/len(train_indices), **evaluate())
        history.append(row)
        if row['validation_selection_score'] < best:
            best = row['validation_selection_score']
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
            model.export(checkpoint, args.inference_dtype)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    exported = BoundarySegmenter(checkpoint)
    verification = evaluate(exported)
    if verification['export_max_abs_error'] > 1e-5 or abs(verification['validation_selection_score']-best) > 1e-5:
        failure = dict(status='export_verification_failed', checkpoint_sha256=exported.sha256,
            trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            data_sha256=data_sha256, selected_torch_score=best, tolerance=1e-5,
            history=history, **verification)
        (args.out/'export-failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise ValueError('exported NumPy model differs from trained convolution')
    if command_history:
        assert hashlib.sha256(args.previous_command_masks.read_bytes()).hexdigest() == command_history['sha256']
    if critical_training:
        assert hashlib.sha256(args.critical_material_labels.read_bytes()).hexdigest() == critical_training['sha256']
    manifest = dict(status='complete', architecture=f'{frames} observed images' +
        (' plus commanded mask' if args.include_expected_mask else '') +
        (' plus preceding commanded mask' if args.previous_command_masks else '') +
        ' -> 9x9 conv64 -> 1x1 conv32 -> 16 tanh heads -> pixel shuffle4',
        input_frames=frames, include_expected_mask=args.include_expected_mask,
        include_previous_expected_mask=bool(args.previous_command_masks), command_history=command_history,
        inference_dtype=args.inference_dtype,
        input_scales=[1], output_upsample=scale,
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        training_groups=sorted(set(groups[~split].tolist())), validation_groups=sorted(set(groups[split].tolist())),
        train_rows=len(train_indices), validation_rows=len(val_indices), device='cpu', torch=torch.__version__,
        seconds=time.time()-started, data_sha256=data_sha256, sampling=sampling_metadata,
        critical_material_training=critical_training,
        checkpoint_sha256=exported.sha256, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization_sha256=hashlib.sha256(initial_arrays).hexdigest(), history=history, **verification,
        selection_metric='optionally asymmetric weighted smooth-L1 plus native distance-gradient and equal-weight Dice/clDice losses',
        distance_objective=dict(beta_mm=.05, underprediction_weight=args.underprediction_weight,
                                overprediction_weight=1., calibrated_probability=False),
        topology_reference='https://arxiv.org/abs/2003.07311',
        limitation='Dense tile validation is not whole-image topology or closed-loop recovery evidence; no runtime repair')
    (args.out/'training.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ['checkpoint_sha256', *verification]}), flush=True)


if __name__ == '__main__':
    main()
