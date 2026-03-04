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


def _multipolygon_to_coco_segmentation(
    multipolygon: MultiPolygon,
) -> COCOInstanceSegmentationMultiPolygon:
    """Convert MultiPolygon to COCO segmentation."""
    coco_segmentation = []
    for polygon in multipolygon.polygons:
        coco_segmentation.append([x for point in polygon for x in point])
    return coco_segmentation


def _binary_mask_rle_to_coco_segmentation(
    binary_mask_rle: BinaryMaskSegmentation,
) -> COCOInstanceSegmentationRLE:
    binary_mask = binary_mask_rle.get_binary_mask()
    counts = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
    return {"counts": counts, "size": [binary_mask_rle.height, binary_mask_rle.width]}


def get_coco_segmentation(
    segmentation: BinaryMaskSegmentation | MultiPolygon,
) -> tuple[
    COCOInstanceSegmentationRLE | COCOInstanceSegmentationMultiPolygon,
    List[float],
    bool,
]:
    """Returns coco segmentation, bbox in xywh format and iscrowd flag for the given segmentation."""
    if isinstance(segmentation, BinaryMaskSegmentation):
        return (
            _binary_mask_rle_to_coco_segmentation(segmentation),
            segmentation.bounding_box.to_format(BoundingBoxFormat.XYWH),
            True,
        )
    elif isinstance(segmentation, MultiPolygon):
        return (
            _multipolygon_to_coco_segmentation(segmentation),
            segmentation.bounding_box().to_format(BoundingBoxFormat.XYWH),
            False,
        )
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")
