"""Train a compact temporal predictor on whole-trajectory-separated data.

Run with an existing Torch environment. Runtime inference needs only NumPy.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.models.temporal import FEATURE_VERSION, TemporalModel, analytical_prediction, canonical_frame, rotate_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--equivariant', action='store_true', help='learn dynamics in an observed-motion coordinate frame')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as data:
        x, y, split, group = (data[k] for k in ('x', 'y', 'validation', 'group'))
    data_manifest_path = args.data.parent / 'manifest.json'
    data_manifest = json.loads(data_manifest_path.read_text()) if data_manifest_path.exists() else {}
    measurement_source = data_manifest.get('boundary_checkpoint_sha256', 'classical')
    if set(group[~split]) & set(group[split]):
        raise ValueError('trajectory leakage between training and validation')
    if not split.any() or split.all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid data or split')
    raw_x, raw_y = x, y
    if args.equivariant:
        x, direction = canonical_frame(raw_x)
        y = rotate_targets(raw_y, direction)
    mean, scale = x[~split].mean(0), np.maximum(x[~split].std(0), .05)
    xtrain, ytrain = torch.from_numpy(((x[~split] - mean) / scale).astype('float32')), torch.from_numpy(y[~split])
    last_train = torch.from_numpy(x[~split, 6:8] * 5.)
    xval = torch.from_numpy(((x[split] - mean) / scale).astype('float32'))
    last_val, yval = torch.from_numpy(x[split, 6:8] * 5.), torch.from_numpy(y[split])
    active_train = torch.from_numpy(np.any(x[~split, :16] != 0, axis=1)[:, None]) if args.equivariant else 1.
    active_val = torch.from_numpy(np.any(x[split, :16] != 0, axis=1)[:, None]) if args.equivariant else 1.
    model = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    baseline = float(np.linalg.norm(analytical_prediction(raw_x[split]) - raw_y[split], axis=1).mean())
    persistence = float(np.linalg.norm(raw_x[split, 6:8] * 5. - raw_y[split], axis=1).mean())
    print(json.dumps(dict(train=len(xtrain), validation=len(xval), analytical_mm=baseline,
                         persistence_mm=persistence)), flush=True)
    best, history, started = float('inf'), [], time.time()
    checkpoint = args.out / 'temporal.npz'
    for epoch in range(args.epochs):
        model.train()
        indices = rng.permutation(len(xtrain))
        total = 0.
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start:start + args.batch_size]
            gate = active_train[idx] if args.equivariant else 1.
            prediction = last_train[idx] + 3 * torch.tanh(model(xtrain[idx]) / 3) * gate
            loss = torch.nn.functional.smooth_l1_loss(prediction, ytrain[idx], beta=.2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            predicted = last_val + 3 * torch.tanh(model(xval) / 3) * active_val
            error = float(torch.linalg.vector_norm(predicted - yval, dim=1).mean())
        row = dict(epoch=epoch + 1, train_loss=total / len(xtrain), validation_mm=error)
        history.append(row)
        if error < best:
            best = error
            arrays = dict(feature_version=np.array(FEATURE_VERSION), mean=mean, scale=scale,
                          measurement_source=np.array(measurement_source),
                          equivariant=np.array(args.equivariant),
                          supports_calibration=np.array(bool(np.any(x[~split, 16] == 0))))
            for i, layer in enumerate([model[0], model[2], model[4]]):
                arrays[f'w{i}'] = layer.weight.detach().numpy()
                arrays[f'b{i}'] = layer.bias.detach().numpy()
            np.savez_compressed(checkpoint, **arrays)
        print(json.dumps(row), flush=True)
    exported = TemporalModel(checkpoint)
    prediction = exported(raw_x[split])
    observed = float(np.linalg.norm(prediction - raw_y[split], axis=1).mean())
    if abs(observed - best) > 1e-5:
        raise AssertionError('NumPy export differs from trained Torch checkpoint')
    manifest = dict(architecture='18-64-64-2 ReLU; bounded residual XY output', feature_version=FEATURE_VERSION,
        seed=args.seed, epochs=args.epochs, torch=torch.__version__, device='cpu', seconds=time.time()-started,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), checkpoint_sha256=exported.sha256,
        training_groups=sorted(set(group[~split].tolist())), validation_groups=sorted(set(group[split].tolist())),
        train_rows=len(xtrain), validation_rows=len(xval), analytical_mm=baseline,
        persistence_mm=persistence, neural_mm=observed, history=history,
        supports_calibration=exported.supports_calibration,
        measurement_source=measurement_source,
        equivariant=args.equivariant, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='held-out forecast validation is not closed-loop recovery evidence')
    (args.out / 'training.json').write_text(json.dumps(manifest, indent=2))
    print('Exported', checkpoint, 'validation mm', observed, flush=True)


if __name__ == '__main__':
    main()
