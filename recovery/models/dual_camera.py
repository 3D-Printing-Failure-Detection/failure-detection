"""Two camera histories, optionally sharing specified solid-base measurements."""
from .boundary import BoundarySegmenter
from .classical import ClassicalSegmenter


class DualCameraSegmenter:
    measurement_source = 'classical'

    def __init__(self, checkpoint, classical_base_layers=()):
        self.primary = ClassicalSegmenter()
        self.alternative = BoundarySegmenter(checkpoint)
        self.sha256 = self.alternative.sha256
        self.alternative_region = None
        self.frames = 0
        if any(type(i) is not int or i < 0 for i in classical_base_layers):
            raise ValueError('classical base layers must be nonnegative indices')
        self.classical_base_layers = frozenset(classical_base_layers)
        self.classical_base_frames = 0
        self.alternative_description = ('learned walls with classical base'
                                        if self.classical_base_layers else 'learned camera')

    @property
    def learned_frames(self):
        return self.frames-self.classical_base_frames

    @property
    def queries(self):
        return self.alternative.queries

    def measure(self, image, expected_mask=None):
        primary = self.primary.measure(image, expected_mask)
        learned = self.alternative.measure(image, expected_mask)
        # The specification fixes the base layer indices before printing. Both
        # models still observe every frame, preserving the learned image history.
        use_classical = self.frames in self.classical_base_layers
        self.alternative_region = primary if use_classical else learned
        self.classical_base_frames += int(use_classical)
        self.frames += 1
        return primary
