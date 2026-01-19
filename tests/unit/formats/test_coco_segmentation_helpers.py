import numpy as np
import pytest

import labelformat.formats.coco_segmentation_helpers as coco_segmentation_helpers
from labelformat.formats.coco_segmentation_helpers import (
    COCOInstanceSegmentationMultiPolygon,
    COCOInstanceSegmentationRLE,
)
from labelformat.model.binary_mask_segmentation import RLEDecoderEncoder
from labelformat.model.bounding_box import BoundingBox
from labelformat.types import ParseError


def test_coco_segmentation_to_binary_mask_rle_roundtrip() -> None:
    mask = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int_)
    counts = RLEDecoderEncoder.encode_column_wise_rle(binary_mask=mask)
    segmentation: COCOInstanceSegmentationRLE = {
        "counts": counts,
        "size": [mask.shape[0], mask.shape[1]],
    }
    bbox = [1.0, 2.0, 3.0, 4.0]

    result = coco_segmentation_helpers.coco_segmentation_to_binary_mask_rle(
        segmentation=segmentation, bbox=bbox
    )

    np.testing.assert_array_equal(result.get_binary_mask(), mask)
    assert result.bounding_box == BoundingBox(xmin=1.0, ymin=2.0, xmax=4.0, ymax=6.0)


def test_coco_segmentation_to_multipolygon() -> None:
    coco_segmentation: COCOInstanceSegmentationMultiPolygon = [
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2.5, 2, 3.5, 2, 3.5, 3, 2.5, 3],
    ]

    multipolygon = coco_segmentation_helpers.coco_segmentation_to_multipolygon(
        coco_segmentation=coco_segmentation
    )

    assert multipolygon.polygons == [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [(2.5, 2.0), (3.5, 2.0), (3.5, 3.0), (2.5, 3.0)],
    ]


def test_coco_segmentation_to_multipolygon_rejects_odd_length() -> None:
    coco_segmentation: COCOInstanceSegmentationMultiPolygon = [[0, 0, 1, 0, 1]]

    with pytest.raises(ParseError, match="Invalid polygon with 5 points"):
        coco_segmentation_helpers.coco_segmentation_to_multipolygon(
            coco_segmentation=coco_segmentation
        )
