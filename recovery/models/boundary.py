"""Learned local signed-distance segmentation at the camera's full resolution.

The small network consumes grayscale camera patches, optionally including the
preceding observed image. NumPy/OpenCV inference uses exported trained weights,
with no Torch or simulator access at runtime.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np

from .. import camera

PATCH_SIZE = 9


def validate_input_scales(scales):
    scales = tuple(scales)
    if not scales or len(set(scales)) != len(scales) or any(s not in (1, 2, 4) for s in scales):
        raise ValueError('invalid boundary input scales')
    return tuple(int(s) for s in scales)


def camera_patches(gray, pixels, input_scales=(1,)):
    """Normalized grayscale samples at fine and wider-context pixel strides."""
    scales = validate_input_scales(input_scales)
    pixels = np.asarray(pixels, dtype=int).reshape(-1, 2)
    radius = PATCH_SIZE // 2
    pad = radius * max(scales)
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    xx, yy = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    return np.concatenate([padded[pixels[:, 1, None] + yy.ravel()*s + pad,
                                  pixels[:, 0, None] + xx.ravel()*s + pad] for s in scales], axis=1)


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
            self.distance_quantile = float(data['distance_quantile']) if 'distance_quantile' in data else None
            self.query_weights = data['query_weights'] if 'query_weights' in data else None
            self.input_scales = validate_input_scales(data['input_scales'] if 'input_scales' in data else (1,))
            self.input_frames = int(data['input_frames']) if 'input_frames' in data else 1
            self.weights = [(data[f'w{i}'], data[f'b{i}']) for i in range(3)]
        if self.input_frames not in (1, 2):
            raise ValueError('boundary input frames must be one or two')
        if self.distance_quantile is not None and not 0 < self.distance_quantile < 1:
            raise ValueError('invalid boundary distance quantile')
        subpixel_offsets(self.output_upsample)
        width = PATCH_SIZE ** 2 * len(self.input_scales) * self.input_frames
        for w, b in self.weights:
            if w.ndim != 2 or w.shape[1] != width or b.shape != (len(w),):
                raise ValueError('invalid boundary network shape')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('non-finite boundary weights')
            width = len(w)
        if self.query_weights is not None:
            if (self.query_weights.shape != (len(self.weights[1][1]), 2)
                    or not np.isfinite(self.query_weights).all() or self.output_upsample == 1):
                raise ValueError('invalid subpixel query weights')
        outputs = 1 if self.query_weights is not None else self.output_upsample**2
        if (width != outputs or upsample < self.output_upsample
                or upsample % self.output_upsample):
            raise ValueError('invalid boundary output or upsample factor')
        self.upsample = upsample
        self._sampling_grid = None
        self._previous_gray = None
        self.queries = 0

    def reset_history(self):
        self._previous_gray = None

    def prime_history(self, previous_image):
        """Supply the actual preceding camera frame for a nonsequential replay."""
        self._previous_gray = normalized_gray(previous_image)

    def _distance_outputs(self, hidden):
        weights, bias = self.weights[1]
        hidden = hidden @ weights.T + bias
        weights, bias = self.weights[2]
        if self.query_weights is not None:
            # All subpixel locations share one distance decoder. Conditioning
            # the hidden activations on XY represents a continuous function
            # inside each pixel instead of sixteen unrelated output heads.
            values = [np.tanh(np.maximum(hidden + self.query_weights @ offset, 0)
                              @ weights.T + bias)[..., 0]
                      for offset in subpixel_offsets(self.output_upsample)]
            return np.stack(values, axis=-1)
        return np.tanh(np.maximum(hidden, 0) @ weights.T + bias)

    def predict_patches(self, patches):
        """Export verification on normalized patches, without camera state."""
        weights, bias = self.weights[0]
        return self._distance_outputs(np.maximum(patches @ weights.T + bias, 0))

    def signed_distance(self, image):
        gray = normalized_gray(image)
        frames = [gray]
        if self.input_frames == 2:
            previous = gray if self._previous_gray is None else self._previous_gray
            if previous.shape != gray.shape:
                raise ValueError('previous camera frame has a different shape; reset the history')
            frames.append(previous)
        weights, bias = self.weights[0]
        channels = []
        for kernels, offset in zip(weights.reshape(len(weights), self.input_frames,
                                                   len(self.input_scales), PATCH_SIZE, PATCH_SIZE), bias):
            channel = np.full_like(gray, offset)
            for frame, frame_kernels in zip(frames, kernels):
                for kernel, scale in zip(frame_kernels, self.input_scales):
                    if scale > 1:
                        sparse = np.zeros(((PATCH_SIZE-1)*scale+1,)*2, dtype='float32')
                        sparse[::scale, ::scale] = kernel
                        kernel = sparse
                    channel += cv2.filter2D(frame, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
            channels.append(channel)
        hidden = np.stack(channels, axis=-1)
        hidden = np.maximum(hidden, 0)
        distance = self._distance_outputs(hidden)
        scale = self.output_upsample
        height, width = gray.shape
        # Each output distance represents its own position within a camera pixel.
        # Interpolating a single value per pixel cannot represent a narrow
        # positive region when every surrounding pixel centre lies outside it.
        return distance.reshape(height, width, scale, scale).transpose(
            0, 2, 1, 3).reshape(height*scale, width*scale)

    def measure(self, image, expected_mask=None):
        self.queries += 1
        distance = self.signed_distance(image)
        if self.input_frames == 2:
            self._previous_gray = normalized_gray(image)
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
