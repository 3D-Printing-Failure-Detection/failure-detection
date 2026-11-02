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


def subpixel_offsets(scale):
    """Image XY offsets of native output samples, with no half-pixel shift."""
    if scale not in (1, 2, 4):
        raise ValueError('unsupported native boundary output scale')
    x, y = np.meshgrid(np.arange(scale, dtype='float32') / scale,
                       np.arange(scale, dtype='float32') / scale)
    return np.column_stack([x.ravel(), y.ravel()])


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
            self.output_upsample = int(data['output_upsample']) if 'output_upsample' in data else 1
            self.weights = [(data[f'w{i}'], data[f'b{i}']) for i in range(3)]
        subpixel_offsets(self.output_upsample)
        width = PATCH_SIZE ** 2
        for w, b in self.weights:
            if w.ndim != 2 or w.shape[1] != width or b.shape != (len(w),):
                raise ValueError('invalid boundary network shape')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('non-finite boundary weights')
            width = len(w)
        if (width != self.output_upsample**2 or upsample < self.output_upsample
                or upsample % self.output_upsample):
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
        scale = self.output_upsample
        height, width = gray.shape
        # Each output head predicts its own position within a camera pixel.
        # Interpolating a single value per pixel cannot represent a narrow
        # positive region when every surrounding pixel centre lies outside it.
        return np.tanh(hidden).reshape(height, width, scale, scale).transpose(
            0, 2, 1, 3).reshape(height*scale, width*scale)

    def measure(self, image, expected_mask=None):
        self.queries += 1
        distance = self.signed_distance(image)
        if self.upsample > self.output_upsample:
            factor = self.upsample // self.output_upsample
            height, width = distance.shape
            if self._sampling_grid is None or self._sampling_grid[0].shape != (height*factor, width*factor):
                # Preserve the original pixel-centre coordinates. resize()'s
                # half-pixel offset would otherwise translate the measured part.
                self._sampling_grid = np.meshgrid(np.arange(width*factor, dtype='float32') / factor,
                                                 np.arange(height*factor, dtype='float32') / factor)
            distance = cv2.remap(distance, *self._sampling_grid, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        mask = (distance > 0).astype('uint8')
        cfg = camera.C.CameraCfg(px_per_mm=camera.CAM.px_per_mm * self.upsample,
                                  fov_mm=camera.CAM.fov_mm)
        return camera.mask_to_region(mask, cfg, min_area=4., simplify=.03)
