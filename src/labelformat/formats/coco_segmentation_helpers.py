from __future__ import annotations

from typing import List, TypedDict

from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.multipolygon import MultiPolygon
from labelformat.types import ParseError


class COCOInstanceSegmentationRLE(TypedDict):
    counts: list[int]
    size: list[int]


COCOInstanceSegmentationMultiPolygon = List[List[float]]


def coco_segmentation_to_binary_mask_rle(
    segmentation: COCOInstanceSegmentationRLE, bbox: list[float]
) -> BinaryMaskSegmentation:
    counts = segmentation["counts"]
    height, width = segmentation["size"]
    binary_mask = RLEDecoderEncoder.decode_column_wise_rle(
        rle=counts, height=height, width=width
    )
    bounding_box = BoundingBox.from_format(bbox=bbox, format=BoundingBoxFormat.XYWH)
    return BinaryMaskSegmentation.from_binary_mask(
        binary_mask=binary_mask, bounding_box=bounding_box
    )


def coco_segmentation_to_multipolygon(
    coco_segmentation: COCOInstanceSegmentationMultiPolygon,
) -> MultiPolygon:
    """Convert COCO segmentation to MultiPolygon."""
    polygons = []
    for polygon in coco_segmentation:
        if len(polygon) % 2 != 0:
            raise ParseError(f"Invalid polygon with {len(polygon)} points: {polygon}")
        polygons.append(
            list(
                zip(
                    [float(x) for x in polygon[0::2]],
                    [float(x) for x in polygon[1::2]],
                )
            )
        )
    return MultiPolygon(polygons=polygons)
