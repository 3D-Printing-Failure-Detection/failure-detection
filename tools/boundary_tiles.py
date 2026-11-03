"""Dense camera training crops in the runtime's pixel-centre coordinates."""
import cv2
import numpy as np

from recovery.models.boundary import PATCH_SIZE


def tile_inputs(frames, origin_xy, size):
    """The valid 9x9 convolution covers size-by-size output pixel centres."""
    x, y = map(int, origin_xy)
    radius = PATCH_SIZE // 2
    if size < 1 or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError('invalid tile size or unequal frame shapes')
    height, width = frames[0].shape
    if not (0 <= x <= width-size and 0 <= y <= height-size):
        raise ValueError('tile output window is outside the image')
    return np.stack([cv2.copyMakeBorder(frame, radius, radius, radius, radius,
        cv2.BORDER_REFLECT_101)[y:y+size+2*radius, x:x+size+2*radius] for frame in frames])


def tile_query_pixels(origin_xy, size, scale):
    x, y = np.meshgrid(np.arange(size*scale, dtype='float32')/scale,
                       np.arange(size*scale, dtype='float32')/scale)
    return np.stack([x, y], axis=-1) + np.asarray(origin_xy)
