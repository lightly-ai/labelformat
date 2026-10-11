from __future__ import annotations

import pytest

from labelformat.model.category import Category
from labelformat.model.instance_segmentation import SingleInstanceSegmentation
from labelformat.model.multipolygon import MultiPolygon

_SQUARE = MultiPolygon(polygons=[[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]])


class TestSingleInstanceSegmentation:
    @pytest.mark.parametrize("confidence", [None, 0.0, 1.0])
    def test_confidence_valid(self, confidence: float | None) -> None:
        instance = SingleInstanceSegmentation(
            category=Category(id=0, name="cat"),
            segmentation=_SQUARE,
            confidence=confidence,
        )
        assert instance.confidence == confidence

    @pytest.mark.parametrize(
        "confidence, expected_error",
        [
            (-0.1, "Confidence must be between 0 and 1, but got: -0.1"),
            (1.1, "Confidence must be between 0 and 1, but got: 1.1"),
        ],
    )
    def test_confidence_out_of_bounds(
        self, confidence: float, expected_error: str
    ) -> None:
        with pytest.raises(ValueError, match=expected_error):
            SingleInstanceSegmentation(
                category=Category(id=0, name="cat"),
                segmentation=_SQUARE,
                confidence=confidence,
            )
