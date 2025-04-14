from __future__ import annotations

import numpy as np
import pytest

from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.object_detection import SingleObjectDetection


class TestSingleObjectDetection:

    @pytest.mark.parametrize(
        "confidence",
        [
            None,
            0.0,
            1.0,
        ],
    )
    def test_confidence_valid(self, confidence: float | None) -> None:
        detection = SingleObjectDetection(
            category=Category(id=0, name="cat"),
            box=BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1),
            confidence=confidence,
        )
        assert detection.confidence == confidence

    @pytest.mark.parametrize(
        "confidence, expected_error",
        [
            (-0.1, "Confidence must be between 0 and 1"),
            (1.1, "Confidence must be between 0 and 1"),
        ],
    )
    def test_confidence_out_of_bounds(
        self, confidence: float, expected_error: str
    ) -> None:
        with pytest.raises(ValueError, match=expected_error):
            SingleObjectDetection(
                category=Category(id=0, name="cat"),
                box=BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1),
                confidence=confidence,
            )
