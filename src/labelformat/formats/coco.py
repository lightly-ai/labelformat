from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, TypedDict

from labelformat.cli.registry import Task, cli_register
from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    InstanceSegmentationOutput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    SingleObjectDetection,
)
from labelformat.types import JsonDict, ParseError


class _COCOBaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to input COCO JSON file",
        )

    def __init__(self, input_file: Path) -> None:
        with input_file.open() as file:
            self._data = json.load(file)

    def get_categories(self) -> Iterable[Category]:
        for category in self._data["categories"]:
            yield Category(
                id=category["id"],
                name=category["name"],
            )

    def get_images(self) -> Iterable[Image]:
        for image in self._data["images"]:
            yield Image(
                id=image["id"],
                filename=image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
            )


class _COCOInstanceSegmentationRLE(TypedDict):
    counts: list[int]
    size: list[int]


_COCOInstanceSegmentationMultiPolygon = List[List[float]]


@cli_register(format="coco", task=Task.OBJECT_DETECTION)
class COCOObjectDetectionInput(_COCOBaseInput, ObjectDetectionInput):
    def get_labels(self) -> Iterable[ImageObjectDetection]:
        image_id_to_image = {image.id: image for image in self.get_images()}
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        image_id_to_annotations: Dict[int, List[JsonDict]] = {
            image_id: [] for image_id in image_id_to_image.keys()
        }
        for ann in self._data["annotations"]:
            image_id_to_annotations[ann["image_id"]].append(ann)

        for image_id, annotations in image_id_to_annotations.items():
            objects = []
            for ann in annotations:
                objects.append(
                    SingleObjectDetection(
                        category=category_id_to_category[ann["category_id"]],
                        box=BoundingBox.from_format(
                            bbox=[float(x) for x in ann["bbox"]],
                            format=BoundingBoxFormat.XYWH,
                        ),
                    )
                )
            yield ImageObjectDetection(
                image=image_id_to_image[image_id],
                objects=objects,
            )


@cli_register(format="coco", task=Task.INSTANCE_SEGMENTATION)
class COCOInstanceSegmentationInput(_COCOBaseInput, InstanceSegmentationInput):
    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        image_id_to_image = {image.id: image for image in self.get_images()}
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        image_id_to_annotations: Dict[int, List[JsonDict]] = {
            image_id: [] for image_id in image_id_to_image.keys()
        }
        for ann in self._data["annotations"]:
            image_id_to_annotations[ann["image_id"]].append(ann)

        for image_id, annotations in image_id_to_annotations.items():
            objects = []
            for ann in annotations:
                if "segmentation" not in ann:
                    raise ParseError(f"Segmentation missing for image id {image_id}")
                segmentation: MultiPolygon | BinaryMaskSegmentation
                if ann["iscrowd"] == 1:
                    segmentation = _coco_segmentation_to_binary_mask_rle(
                        segmentation=ann["segmentation"], bbox=ann["bbox"]
                    )
                else:
                    segmentation = _coco_segmentation_to_multipolygon(
                        coco_segmentation=ann["segmentation"]
                    )
                objects.append(
                    SingleInstanceSegmentation(
                        category=category_id_to_category[ann["category_id"]],
                        segmentation=segmentation,
                    )
                )
            yield ImageInstanceSegmentation(
                image=image_id_to_image[image_id],
                objects=objects,
            )


class _COCOBaseOutput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-file",
            type=Path,
            required=True,
            help="Path to output COCO JSON file",
        )

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file


@cli_register(format="coco", task=Task.OBJECT_DETECTION)
class COCOObjectDetectionOutput(_COCOBaseOutput, ObjectDetectionOutput):
    def save(self, label_input: ObjectDetectionInput) -> None:
        data = {}
        data["images"] = _get_output_images_dict(images=label_input.get_images())
        data["categories"] = _get_output_categories_dict(
            categories=label_input.get_categories()
        )
        data["annotations"] = []
        for label in label_input.get_labels():
            for obj in label.objects:
                annotation = {
                    "image_id": label.image.id,
                    "category_id": obj.category.id,
                    "bbox": [
                        float(v) for v in obj.box.to_format(BoundingBoxFormat.XYWH)
                    ],
                }
                data["annotations"].append(annotation)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w") as file:
            json.dump(data, file, indent=2)


@cli_register(format="coco", task=Task.INSTANCE_SEGMENTATION)
class COCOInstanceSegmentationOutput(_COCOBaseOutput, InstanceSegmentationOutput):
    def save(self, label_input: InstanceSegmentationInput) -> None:
        data = {}
        data["images"] = _get_output_images_dict(images=label_input.get_images())
        data["categories"] = _get_output_categories_dict(
            categories=label_input.get_categories()
        )
        data["annotations"] = []
        for label in label_input.get_labels():
            for obj in label.objects:
                segmentation: (
                    _COCOInstanceSegmentationMultiPolygon | _COCOInstanceSegmentationRLE
                )
                if isinstance(obj.segmentation, BinaryMaskSegmentation):
                    segmentation = _binary_mask_rle_to_coco_segmentation(
                        binary_mask_rle=obj.segmentation
                    )
                    bbox = [
                        float(v)
                        for v in obj.segmentation.bounding_box.to_format(
                            BoundingBoxFormat.XYWH
                        )
                    ]
                    iscrowd = 1
                elif isinstance(obj.segmentation, MultiPolygon):
                    segmentation = _multipolygon_to_coco_segmentation(
                        multipolygon=obj.segmentation
                    )
                    bbox = [
                        float(v)
                        for v in obj.segmentation.bounding_box().to_format(
                            BoundingBoxFormat.XYWH
                        )
                    ]
                    iscrowd = 0
                else:
                    raise ParseError(
                        f"Unsupported segmentation type: {type(obj.segmentation)}"
                    )
                annotation = {
                    "image_id": label.image.id,
                    "category_id": obj.category.id,
                    "bbox": bbox,
                    "iscrowd": iscrowd,
                    "segmentation": segmentation,
                }
                data["annotations"].append(annotation)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w") as file:
            json.dump(data, file, indent=2)


def _coco_segmentation_to_multipolygon(
    coco_segmentation: _COCOInstanceSegmentationMultiPolygon,
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
) -> _COCOInstanceSegmentationMultiPolygon:
    """Convert MultiPolygon to COCO segmentation."""
    coco_segmentation = []
    for polygon in multipolygon.polygons:
        coco_segmentation.append([x for point in polygon for x in point])
    return coco_segmentation


def _coco_segmentation_to_binary_mask_rle(
    segmentation: _COCOInstanceSegmentationRLE, bbox: list[float]
) -> BinaryMaskSegmentation:
    counts = segmentation["counts"]
    height, width = segmentation["size"]
    binary_mask = RLEDecoderEncoder.decode_column_wise_rle(counts, height, width)
    bounding_box = BoundingBox.from_format(bbox=bbox, format=BoundingBoxFormat.XYWH)
    return BinaryMaskSegmentation.from_binary_mask(
        binary_mask, bounding_box=bounding_box
    )


def _binary_mask_rle_to_coco_segmentation(
    binary_mask_rle: BinaryMaskSegmentation,
) -> _COCOInstanceSegmentationRLE:
    binary_mask = binary_mask_rle.get_binary_mask()
    counts = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
    return {"counts": counts, "size": [binary_mask_rle.height, binary_mask_rle.width]}


def _get_output_images_dict(
    images: Iterable[Image],
) -> List[JsonDict]:
    """Get the "images" dict for COCO JSON."""
    return [
        {
            "id": image.id,
            "file_name": image.filename,
            "width": image.width,
            "height": image.height,
        }
        for image in images
    ]


def _get_output_categories_dict(
    categories: Iterable[Category],
) -> List[JsonDict]:
    """Get the "categories" dict for COCO JSON."""
    return [
        {
            "id": category.id,
            "name": category.name,
        }
        for category in categories
    ]
