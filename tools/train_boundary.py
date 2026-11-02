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
from recovery.models.boundary import PATCH_SIZE, BoundarySegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--hidden', type=int, nargs=2, default=[32, 16], metavar=('FIRST', 'SECOND'))
    parser.add_argument('--augment-dihedral', action='store_true',
                        help='random rotations/reflections of training patches; signed distance is invariant')
    args = parser.parse_args()
    if min(args.hidden) < 1:
        parser.error('hidden widths must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        x, y, split, group = [data[k] for k in ('x', 'y', 'validation', 'group')]
    if set(group[~split]) & set(group[split]) or not split.any() or split.all():
        raise ValueError('invalid trajectory-separated split')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite boundary training data')
    args.out.mkdir(parents=True, exist_ok=True)
    xtrain, ytrain = torch.from_numpy(x[~split].astype('float32')), torch.from_numpy(y[~split, None])
    xval, yval = torch.from_numpy(x[split].astype('float32')), torch.from_numpy(y[split, None])
    edge = torch.abs(yval[:, 0]) < .5
    h1, h2 = args.hidden
    model = nn.Sequential(nn.Linear(PATCH_SIZE**2, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))
    grid = np.arange(PATCH_SIZE**2).reshape(PATCH_SIZE, PATCH_SIZE)
    transforms = torch.from_numpy(np.stack([np.rot90(np.fliplr(grid) if flip else grid, rotation).ravel().copy()
                                           for flip in (False, True) for rotation in range(4)]))
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
            loss = (torch.nn.functional.smooth_l1_loss(prediction, ytrain[idx], beta=.05, reduction='none') * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            predicted = torch.tanh(model(xval))
            error = float(torch.abs(predicted[edge] - yval[edge]).mean())
            accuracy = float(((predicted > 0) == (yval > 0)).float().mean())
        row = dict(epoch=epoch+1, train_loss=total/len(xtrain), validation_boundary_mae_mm=error,
                   validation_pixel_accuracy=accuracy)
        history.append(row)
        if error < best:
            best = error
            arrays = dict(patch_size=np.array(PATCH_SIZE))
            for i, layer in enumerate((model[0], model[2], model[4])):
                arrays[f'w{i}'], arrays[f'b{i}'] = layer.weight.detach().numpy(), layer.bias.detach().numpy()
            np.savez_compressed(checkpoint, **arrays)
        print(json.dumps(row), flush=True)
    exported = BoundarySegmenter(checkpoint)
    predicted = x[split].astype('float32')
    for i, (w, b) in enumerate(exported.weights):
        predicted = predicted @ w.T + b
        if i < 2:
            predicted = np.maximum(predicted, 0)
    predicted = np.tanh(predicted[:, 0])
    boundary = np.abs(y[split]) < .5
    error = float(np.abs(predicted[boundary] - y[split][boundary]).mean())
    assert abs(error-best) < 1e-5, 'exported boundary network differs from trained checkpoint'
    manifest = dict(architecture=f'9x9 grayscale patch -> {h1} ReLU -> {h2} ReLU -> tanh signed distance',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        device='cpu', torch=torch.__version__, seconds=time.time()-started, train_rows=len(xtrain),
        validation_rows=len(xval), training_groups=sorted(set(group[~split].tolist())),
        validation_groups=sorted(set(group[split].tolist())), validation_boundary_mae_mm=error,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), checkpoint_sha256=exported.sha256,
        history=history, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        augmentation='random D4 training patches' if args.augment_dihedral else 'none',
        limitation='patch validation is not full-frame or closed-loop recovery validation')
    (args.out / 'training.json').write_text(json.dumps(manifest, indent=2))
    print('Exported', checkpoint, 'boundary MAE mm', error, flush=True)


if __name__ == '__main__':
    main()
