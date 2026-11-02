"""Learned local signed-distance segmentation at the camera's full resolution.

The small network consumes only grayscale camera patches. NumPy/OpenCV inference
uses exported trained weights, with no Torch or simulator access at runtime.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .. import camera

PATCH_SIZE = 9


def normalized_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    reference = max(float(cv2.GaussianBlur(gray, (0, 0), 1.).max()), 1.)
    return gray.astype('float32') / reference


class BoundarySegmenter:
    def __init__(self, checkpoint, upsample=4):
        path = Path(checkpoint)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if int(data['patch_size']) != PATCH_SIZE:
                raise ValueError('unsupported boundary patch size')
            self.weights = [(data[f'w{i}'], data[f'b{i}']) for i in range(3)]
        width = PATCH_SIZE ** 2
        for w, b in self.weights:
            if w.ndim != 2 or w.shape[1] != width or b.shape != (len(w),):
                raise ValueError('invalid boundary network shape')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('non-finite boundary weights')
            width = len(w)
        if width != 1 or upsample < 1:
            raise ValueError('invalid boundary output or upsample factor')
        self.upsample = upsample
        self._sampling_grid = None
        self.queries = 0

    def signed_distance(self, image):
        gray = normalized_gray(image)
        weights, bias = self.weights[0]
        hidden = np.stack([cv2.filter2D(gray, cv2.CV_32F, kernel.reshape(PATCH_SIZE, PATCH_SIZE),
                                       borderType=cv2.BORDER_REFLECT_101) + offset
                           for kernel, offset in zip(weights, bias)], axis=-1)
        hidden = np.maximum(hidden, 0)
        for i, (weights, bias) in enumerate(self.weights[1:]):
            hidden = hidden @ weights.T + bias
            if i == 0:
                hidden = np.maximum(hidden, 0)
        return np.tanh(hidden[..., 0])

    def measure(self, image, expected_mask=None):
        self.queries += 1
        distance = self.signed_distance(image)
        if self.upsample > 1:
            height, width = distance.shape
            if self._sampling_grid is None or self._sampling_grid[0].shape != (height*self.upsample, width*self.upsample):
                # Preserve the original pixel-centre coordinates. resize()'s
                # half-pixel offset would otherwise translate the measured part.
                self._sampling_grid = np.meshgrid(np.arange(width*self.upsample, dtype='float32') / self.upsample,
                                                 np.arange(height*self.upsample, dtype='float32') / self.upsample)
            distance = cv2.remap(distance, *self._sampling_grid, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        mask = (distance > 0).astype('uint8')
        cfg = camera.C.CameraCfg(px_per_mm=camera.CAM.px_per_mm * self.upsample,
                                  fov_mm=camera.CAM.fov_mm)
        return camera.mask_to_region(mask, cfg, min_area=4., simplify=.03)
