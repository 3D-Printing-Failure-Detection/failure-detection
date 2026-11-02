"""Small trained contour forecast, with NumPy inference and explicit provenance.

Inputs contain four camera-derived displacement fields and requested horizon.
Training may use simulator truth as a label; this module cannot access it.
"""
import hashlib
from pathlib import Path

import numpy as np

FEATURE_VERSION = 1


def features(values, translations, ahead, interior):
    """values: four observations x points x XY, translations: four x XY."""
    values = np.asarray(values, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 4 or values.shape[2] != 2:
        raise ValueError('four XY residual observations are required')
    n = values.shape[1]
    if translations.shape != (4, 2) or ahead < 0:
        raise ValueError('invalid translation history or prediction horizon')
    return np.concatenate([values.transpose(1, 0, 2).reshape(n, 8) / 5.,
        np.broadcast_to(translations.reshape(1, 8) / 5., (n, 8)),
        np.full((n, 1), np.log1p(min(ahead, 160)) / 5., dtype=np.float32),
        np.full((n, 1), float(interior), dtype=np.float32)], axis=1)


def analytical_prediction(x):
    values = x[:, :8].reshape(-1, 4, 2) * 5.
    ahead = np.expm1(x[:, 16] * 5.)[:, None]
    trend = np.clip(np.median(np.diff(values, axis=1), axis=1), -.5, .5)
    return values[:, -1] + np.minimum(ahead, 3) * trend


class TemporalModel:
    def __init__(self, checkpoint):
        path = Path(checkpoint)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if int(data['feature_version']) != FEATURE_VERSION:
                raise ValueError('unsupported temporal feature version')
            self.mean, self.scale = data['mean'], data['scale']
            self.weights = [(data[f'w{i}'], data[f'b{i}']) for i in range(3)]
            self.supports_calibration = bool(data['supports_calibration']) if 'supports_calibration' in data else False
        if self.mean.shape != (18,) or self.scale.shape != (18,) or np.any(self.scale <= 0):
            raise ValueError('invalid normalization')
        arrays = [self.mean, self.scale] + [v for pair in self.weights for v in pair]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('non-finite temporal weights')
        width = 18
        for i, (w, b) in enumerate(self.weights):
            if w.ndim != 2 or w.shape[1] != width or b.shape != (w.shape[0],):
                raise ValueError('invalid temporal layer shape')
            width = w.shape[0]
        if width != 2:
            raise ValueError('temporal output must be XY')

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != 18 or not np.isfinite(x).all():
            raise ValueError('invalid camera-derived temporal features')
        hidden = (x - self.mean) / self.scale
        for i, (w, b) in enumerate(self.weights):
            hidden = hidden @ w.T + b
            if i < 2:
                hidden = np.maximum(hidden, 0)
        return x[:, 6:8] * 5. + 3. * np.tanh(hidden / 3.)
