"""Two independent measurements of the same camera frame."""
from .boundary import BoundarySegmenter
from .classical import ClassicalSegmenter


class DualCameraSegmenter:
    measurement_source = 'classical'

    def __init__(self, checkpoint):
        self.primary = ClassicalSegmenter()
        self.alternative = BoundarySegmenter(checkpoint)
        self.sha256 = self.alternative.sha256
        self.alternative_region = None
        self.frames = 0

    @property
    def queries(self):
        return self.alternative.queries

    def measure(self, image, expected_mask=None):
        primary = self.primary.measure(image, expected_mask)
        self.alternative_region = self.alternative.measure(image, expected_mask)
        self.frames += 1
        return primary
