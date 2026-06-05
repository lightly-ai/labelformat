import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import pytest

from labelformat.formats.coco import (
    COCOInstanceSegmentationInput,
    COCOInstanceSegmentationOutput,
    COCOObjectDetectionInput,
    COCOObjectDetectionOutput,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)

from ... import simple_object_detection_label_input


def _create_coco_file(tmp_path: Path, with_score: bool) -> Path:
    annotations = [
        {
            "image_id": 0,
            "category_id": 1,
            "bbox": [10.0, 20.0, 20.0, 20.0],
        },
        {
            "image_id": 0,
            "category_id": 0,
            "bbox": [50.0, 60.0, 20.0, 20.0],
        },
    ]
    if with_score:
        annotations[0]["score"] = 0.4
        annotations[1]["score"] = 0.8
    data = {
        "images": [
            {"id": 0, "file_name": "image.jpg", "width": 100, "height": 200},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cow"},
        ],
        "annotations": annotations,
    }
    coco_file = tmp_path / "train.json"
    coco_file.write_text(json.dumps(data))
    return coco_file


class TestCOCOObjectDetectionInput:
    @pytest.mark.parametrize("with_score", [True, False])
    def test_get_labels(self, tmp_path: Path, with_score: bool) -> None:
        coco_file = _create_coco_file(tmp_path=tmp_path, with_score=with_score)
        label_input = COCOObjectDetectionInput(input_file=coco_file)
        labels = list(label_input.get_labels())
        assert labels == [
            ImageObjectDetection(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleObjectDetection(
                        category=Category(id=1, name="dog"),
                        box=BoundingBox(xmin=10.0, ymin=20.0, xmax=30.0, ymax=40.0),
                        confidence=0.4 if with_score else None,
                    ),
                    SingleObjectDetection(
                        category=Category(id=0, name="cat"),
                        box=BoundingBox(xmin=50.0, ymin=60.0, xmax=70.0, ymax=80.0),
                        confidence=0.8 if with_score else None,
                    ),
                ],
            ),
        ]


class TestCOCOObjectDetectionOutput:
    @pytest.mark.parametrize("with_confidence", [True, False])
    def test_save(self, tmp_path: Path, with_confidence: bool) -> None:
        output_file = tmp_path / "train.json"
        COCOObjectDetectionOutput(output_file=output_file).save(
            label_input=simple_object_detection_label_input.get_input(
                with_confidence=with_confidence
            )
        )

        output_json = json.loads(output_file.read_text())
        expected_annotations = [
            {
                "image_id": 0,
                "category_id": 1,
                "bbox": [10.0, 20.0, 20.0, 20.0],
            },
            {
                "image_id": 0,
                "category_id": 0,
                "bbox": [50.0, 60.0, 20.0, 20.0],
            },
        ]
        if with_confidence:
            expected_annotations[0]["score"] = 0.4
            expected_annotations[1]["score"] = 0.8
        assert output_json["annotations"] == expected_annotations


def _create_coco_instance_segmentation_file(tmp_path: Path, with_score: bool) -> Path:
    annotations = [
        {
            "image_id": 0,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [10.0, 10.0, 10.0, 10.0],
            "segmentation": [[10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 10.0]],
        },
    ]
    if with_score:
        annotations[0]["score"] = 0.4
    data = {
        "images": [
            {"id": 0, "file_name": "image.jpg", "width": 100, "height": 200},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cow"},
        ],
        "annotations": annotations,
    }
    coco_file = tmp_path / "train.json"
    coco_file.write_text(json.dumps(data))
    return coco_file


class TestCOCOInstanceSegmentationInput:
    @pytest.mark.parametrize("with_score", [True, False])
    def test_get_labels(self, tmp_path: Path, with_score: bool) -> None:
        coco_file = _create_coco_instance_segmentation_file(
            tmp_path=tmp_path, with_score=with_score
        )
        label_input = COCOInstanceSegmentationInput(input_file=coco_file)
        labels = list(label_input.get_labels())
        assert labels == [
            ImageInstanceSegmentation(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleInstanceSegmentation(
                        category=Category(id=1, name="dog"),
                        segmentation=MultiPolygon(
                            polygons=[
                                [
                                    (10.0, 10.0),
                                    (10.0, 20.0),
                                    (20.0, 20.0),
                                    (20.0, 10.0),
                                ],
                            ],
                        ),
                        confidence=0.4 if with_score else None,
                    ),
                ],
            ),
        ]


class _SimpleInstanceSegmentationInput(InstanceSegmentationInput):
    def __init__(self, with_confidence: bool) -> None:
        self._with_confidence = with_confidence

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        pass

    def get_categories(self) -> Iterable[Category]:
        return [
            Category(id=0, name="cat"),
            Category(id=1, name="dog"),
            Category(id=2, name="cow"),
        ]

    def get_images(self) -> Iterable[Image]:
        return [Image(id=0, filename="image.jpg", width=100, height=200)]

    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        return [
            ImageInstanceSegmentation(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleInstanceSegmentation(
                        category=Category(id=1, name="dog"),
                        segmentation=MultiPolygon(
                            polygons=[
                                [
                                    (10.0, 10.0),
                                    (10.0, 20.0),
                                    (20.0, 20.0),
                                    (20.0, 10.0),
                                ],
                            ],
                        ),
                        confidence=0.4 if self._with_confidence else None,
                    ),
                ],
            )
        ]


class TestCOCOInstanceSegmentationOutput:
    @pytest.mark.parametrize("with_confidence", [True, False])
    def test_save(self, tmp_path: Path, with_confidence: bool) -> None:
        output_file = tmp_path / "train.json"
        COCOInstanceSegmentationOutput(output_file=output_file).save(
            label_input=_SimpleInstanceSegmentationInput(
                with_confidence=with_confidence
            )
        )

        output_json = json.loads(output_file.read_text())
        annotations = output_json["annotations"]
        assert len(annotations) == 1
        if with_confidence:
            assert annotations[0]["score"] == 0.4
        else:
            assert "score" not in annotations[0]
