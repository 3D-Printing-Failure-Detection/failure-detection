"""Train a tiny camera-patch network with full-resolution OpenCV export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.models.boundary import PATCH_SIZE, BoundarySegmenter, subpixel_offsets, validate_input_scales


class QueryBoundaryNetwork(nn.Module):
    """One distance decoder shared across subpixel query coordinates."""
    def __init__(self, h1, h2, scale, input_width):
        super().__init__()
        self.first, self.second, self.last = nn.Linear(input_width, h1), nn.Linear(h1, h2), nn.Linear(h2, 1)
        self.query = nn.Linear(2, h2, bias=False)
        self.register_buffer('offsets', torch.from_numpy(subpixel_offsets(scale)))

    def forward(self, patches):
        hidden = self.second(torch.relu(self.first(patches)))
        return self.last(torch.relu(hidden[:, None, :] + self.query(self.offsets))).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--extra-data', type=Path, nargs='*', default=[],
                        help='additional whole trajectories with the same patch and output format')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--hidden', type=int, nargs=2, default=[32, 16], metavar=('FIRST', 'SECOND'))
    parser.add_argument('--augment-dihedral', action='store_true',
                        help='random rotations/reflections of training patches; signed distance is invariant')
    parser.add_argument('--distance-quantile', type=float,
                        help='learn a conditional distance quantile with pinball loss instead of a central estimate')
    parser.add_argument('--query-conditioned', action='store_true',
                        help='share one distance decoder across native subpixel XY queries')
    parser.add_argument('--select-input-scales', type=int, nargs='+', choices=[1, 2, 4],
                        help='train a matched ablation using a subset of the saved camera inputs')
    args = parser.parse_args()
    if min(args.hidden) < 1:
        parser.error('hidden widths must be positive')
    if args.distance_quantile is not None and not 0 < args.distance_quantile < 1:
        parser.error('--distance-quantile must be strictly between zero and one')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        x, y, split, group = [data[k] for k in ('x', 'y', 'validation', 'group')]
        output_upsample = int(data['output_upsample']) if 'output_upsample' in data else 1
        input_scales = validate_input_scales(data['input_scales'] if 'input_scales' in data else (1,))
    sources = [dict(path=str(args.data.resolve()), sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), rows=len(x))]
    seen_groups = set(group.tolist())
    for path in args.extra_data:
        with np.load(path, allow_pickle=False) as data:
            xx, yy, ss, gg = [data[k] for k in ('x', 'y', 'validation', 'group')]
            scale = int(data['output_upsample']) if 'output_upsample' in data else 1
            scales = validate_input_scales(data['input_scales'] if 'input_scales' in data else (1,))
        if (scale != output_upsample or scales != input_scales
                or xx.shape[1:] != x.shape[1:] or yy.shape[1:] != y.shape[1:]):
            raise ValueError('additional dataset has incompatible camera inputs or output targets')
        if seen_groups & set(gg.tolist()):
            raise ValueError('additional dataset repeats an existing trajectory')
        seen_groups.update(gg.tolist())
        x, y, split, group = [np.concatenate([a, b]) for a, b in zip((x, y, split, group), (xx, yy, ss, gg))]
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(xx)))
    dataset_input_scales = input_scales
    if args.select_input_scales is not None:
        selected = validate_input_scales(args.select_input_scales)
        if not set(selected).issubset(input_scales):
            parser.error('selected input scales must exist in every dataset')
        columns = np.concatenate([np.arange(PATCH_SIZE**2) + input_scales.index(s)*PATCH_SIZE**2
                                  for s in selected])
        x = x[:, columns]
        input_scales = selected
    if output_upsample > 1 and args.augment_dihedral:
        parser.error('D4 augmentation requires rotating the subpixel targets and is not implemented')
    if args.query_conditioned and output_upsample == 1:
        parser.error('--query-conditioned requires native subpixel targets')
    y = y.reshape(len(x), -1)
    input_width = PATCH_SIZE**2 * len(input_scales)
    if x.ndim != 2 or x.shape[1] != input_width:
        raise ValueError('boundary inputs do not match the declared context scales')
    if y.shape[1] != output_upsample**2:
        raise ValueError('boundary targets do not match the declared output grid')
    if set(group[~split]) & set(group[split]) or not split.any() or split.all():
        raise ValueError('invalid trajectory-separated split')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite boundary training data')
    args.out.mkdir(parents=True, exist_ok=True)
    xtrain, ytrain = torch.from_numpy(x[~split].astype('float32')), torch.from_numpy(y[~split])
    xval, yval = torch.from_numpy(x[split].astype('float32')), torch.from_numpy(y[split])
    edge = torch.abs(yval) < .5
    h1, h2 = args.hidden
    model = (QueryBoundaryNetwork(h1, h2, output_upsample, input_width) if args.query_conditioned else
             nn.Sequential(nn.Linear(input_width, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                           nn.Linear(h2, output_upsample**2)))
    grid = np.arange(PATCH_SIZE**2).reshape(PATCH_SIZE, PATCH_SIZE)
    permutations = [np.rot90(np.fliplr(grid) if flip else grid, rotation).ravel().copy()
                    for flip in (False, True) for rotation in range(4)]
    transforms = torch.from_numpy(np.stack([np.concatenate([p + i*PATCH_SIZE**2 for i in range(len(input_scales))])
                                           for p in permutations]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    best, history, started = float('inf'), [], time.time()
    checkpoint = args.out / 'boundary.npz'
    print('train', len(xtrain), 'validation', len(xval), 'boundary', int(edge.sum()), flush=True)
    for epoch in range(args.epochs):
        indices = rng.permutation(len(xtrain))
        total = 0.
        model.train()
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start:start+args.batch_size]
            inputs = xtrain[idx]
            if args.augment_dihedral:
                permutation = transforms[rng.integers(len(transforms), size=len(idx))]
                inputs = torch.gather(inputs, 1, permutation)
            prediction = torch.tanh(model(inputs))
            weight = .25 + .75 * (torch.abs(ytrain[idx]) < .5)
            if args.distance_quantile is None:
                point_loss = torch.nn.functional.smooth_l1_loss(prediction, ytrain[idx], beta=.05, reduction='none')
            else:
                residual = ytrain[idx] - prediction
                point_loss = torch.maximum(args.distance_quantile * residual,
                                           (args.distance_quantile - 1) * residual)
            loss = (point_loss * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            predicted = torch.tanh(model(xval))
            error = float(torch.abs(predicted[edge] - yval[edge]).mean())
            accuracy = float(((predicted > 0) == (yval > 0)).float().mean())
            score = error
            coverage = float((predicted[edge] >= yval[edge]).float().mean())
            if args.distance_quantile is not None:
                residual = yval - predicted
                point_loss = torch.maximum(args.distance_quantile * residual,
                                           (args.distance_quantile - 1) * residual)
                score = float((point_loss * (.25 + .75 * edge)).mean())
        row = dict(epoch=epoch+1, train_loss=total/len(xtrain), validation_boundary_mae_mm=error,
                   validation_pixel_accuracy=accuracy, validation_selection_score=score,
                   validation_boundary_coverage=coverage)
        history.append(row)
        if score < best:
            best = score
            arrays = dict(patch_size=np.array(PATCH_SIZE), output_upsample=np.array(output_upsample),
                          input_scales=np.array(input_scales))
            if args.distance_quantile is not None:
                arrays['distance_quantile'] = np.array(args.distance_quantile)
            layers = (model.first, model.second, model.last) if args.query_conditioned else (model[0], model[2], model[4])
            if args.query_conditioned:
                arrays['query_weights'] = model.query.weight.detach().numpy()
            for i, layer in enumerate(layers):
                arrays[f'w{i}'], arrays[f'b{i}'] = layer.weight.detach().numpy(), layer.bias.detach().numpy()
            np.savez_compressed(checkpoint, **arrays)
        print(json.dumps(row), flush=True)
    exported = BoundarySegmenter(checkpoint)
    predicted = exported.predict_patches(x[split].astype('float32'))
    boundary = np.abs(y[split]) < .5
    error = float(np.abs(predicted[boundary] - y[split][boundary]).mean())
    score = error
    if args.distance_quantile is not None:
        residual = y[split] - predicted
        point_loss = np.maximum(args.distance_quantile * residual, (args.distance_quantile - 1) * residual)
        score = float((point_loss * (.25 + .75 * boundary)).mean())
    assert abs(score-best) < 1e-5, 'exported boundary network differs from trained checkpoint'
    manifest = dict(architecture=f'9x9 grayscale patch -> {h1} ReLU -> {h2} ReLU -> {output_upsample**2} tanh distances',
        output_upsample=output_upsample,
        input_scales=list(input_scales), input_width=input_width,
        dataset_input_scales=list(dataset_input_scales),
        subpixel_decoder='shared XY-conditioned distance' if args.query_conditioned else 'independent distance heads',
        distance_quantile=args.distance_quantile,
        validation_selection_score=score,
        selection_metric='boundary MAE' if args.distance_quantile is None else 'weighted pinball loss',
        validation_boundary_coverage=float((predicted[boundary] >= y[split][boundary]).mean()),
        args={k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == 'extra_data' else v
              for k, v in vars(args).items()},
        device='cpu', torch=torch.__version__, seconds=time.time()-started, train_rows=len(xtrain),
        validation_rows=len(xval), training_groups=sorted(set(group[~split].tolist())),
        validation_groups=sorted(set(group[split].tolist())), validation_boundary_mae_mm=error,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), checkpoint_sha256=exported.sha256,
        data_sources=sources,
        history=history, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        augmentation='random D4 training patches' if args.augment_dihedral else 'none',
        limitation='patch validation is not full-frame or closed-loop recovery validation')
    if args.query_conditioned:
        manifest['architecture'] = f'9x9 grayscale patch -> {h1} ReLU -> {h2} plus XY query ReLU -> shared tanh distance'
    if len(input_scales) > 1:
        manifest['architecture'] = manifest['architecture'].replace('9x9 grayscale patch',
            f'9x9 grayscale patches at strides {input_scales}')
    (args.out / 'training.json').write_text(json.dumps(manifest, indent=2))
    print('Exported', checkpoint, 'boundary MAE mm', error, flush=True)


if __name__ == '__main__':
    main()
