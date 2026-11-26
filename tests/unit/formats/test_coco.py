import json
from pathlib import Path

import pytest

from labelformat.formats.coco import (
    COCOInstanceSegmentationInput,
    COCOInstanceSegmentationOutput,
    COCOObjectDetectionInput,
    COCOObjectDetectionOutput,
    COCOSingleInstanceSegmentation,
)
from labelformat.formats.labelformat import LabelformatObjectDetectionInput
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)

from ... import simple_object_detection_label_input
from ...simple_instance_segmentation_label_input import (
    SimpleInstanceSegmentationInputWithBinaryMask,
)

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


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
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10.0, 20.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 0,
                "category_id": 0,
                "bbox": [50.0, 60.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            },
        ]
        if with_confidence:
            expected_annotations[0]["score"] = 0.4
            expected_annotations[1]["score"] = 0.8
        assert output_json["annotations"] == expected_annotations

    def test_save__area(self, tmp_path: Path) -> None:
        output_file = tmp_path / "train.json"
        image = Image(id=0, filename="image.jpg", width=100, height=200)
        category = Category(id=0, name="cat")
        label_input = LabelformatObjectDetectionInput(
            categories=[category],
            images=[image],
            labels=[
                ImageObjectDetection(
                    image=image,
                    objects=[
                        SingleObjectDetection(
                            category=category,
                            box=BoundingBox(xmin=10.0, ymin=20.0, xmax=20.0, ymax=50.0),
                        )
                    ],
                )
            ],
        )

        COCOObjectDetectionOutput(output_file=output_file).save(label_input=label_input)

        [annotation] = json.loads(output_file.read_text())["annotations"]
        assert annotation["area"] == 300.0


class TestCOCOInstanceSegmentationInput:
    @pytest.mark.parametrize("iscrowd", [0, 1])
    def test_get_labels__rle(self, tmp_path: Path, iscrowd: int) -> None:
        input_file = _create_coco_instance_segmentation_file(
            tmp_path=tmp_path,
            segmentation={"counts": [2, 4], "size": [2, 3]},
            iscrowd=iscrowd,
        )

        [label] = COCOInstanceSegmentationInput(input_file=input_file).get_labels()

        [obj] = label.objects
        assert isinstance(obj, COCOSingleInstanceSegmentation)
        assert isinstance(obj.segmentation, BinaryMaskSegmentation)
        assert obj.segmentation.get_rle() == [1, 2, 1, 2]
        assert obj.is_crowd == (iscrowd == 1)

    @pytest.mark.parametrize("iscrowd", [0, 1])
    def test_get_labels__polygon(self, tmp_path: Path, iscrowd: int) -> None:
        input_file = _create_coco_instance_segmentation_file(
            tmp_path=tmp_path,
            segmentation=[[1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0]],
            iscrowd=iscrowd,
        )

        [label] = COCOInstanceSegmentationInput(input_file=input_file).get_labels()

        [obj] = label.objects
        assert isinstance(obj, COCOSingleInstanceSegmentation)
        assert isinstance(obj.segmentation, MultiPolygon)
        assert obj.segmentation.polygons == [
            [(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)]
        ]
        assert obj.is_crowd == (iscrowd == 1)

    @pytest.mark.parametrize("iscrowd", [False, True])
    def test_get_labels__boolean_iscrowd(self, tmp_path: Path, iscrowd: bool) -> None:
        # labelformat 0.1.17 and earlier write iscrowd as a JSON boolean.
        input_file = _create_coco_instance_segmentation_file(
            tmp_path=tmp_path,
            segmentation={"counts": [2, 4], "size": [2, 3]},
            iscrowd=iscrowd,
        )

        [label] = COCOInstanceSegmentationInput(input_file=input_file).get_labels()

        [obj] = label.objects
        assert isinstance(obj, COCOSingleInstanceSegmentation)
        assert obj.is_crowd == iscrowd


class TestCOCOInstanceSegmentationOutput:
    @pytest.mark.parametrize("iscrowd", [0, 1])
    @pytest.mark.parametrize(
        "segmentation",
        [
            {"counts": [2, 4], "size": [2, 3]},
            [[1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0]],
        ],
    )
    def test_save__keeps_iscrowd(
        self, tmp_path: Path, segmentation: object, iscrowd: int
    ) -> None:
        input_file = _create_coco_instance_segmentation_file(
            tmp_path=tmp_path, segmentation=segmentation, iscrowd=iscrowd
        )
        output_file = tmp_path / "output.json"

        COCOInstanceSegmentationOutput(output_file=output_file).save(
            label_input=COCOInstanceSegmentationInput(input_file=input_file)
        )

        [annotation] = json.loads(output_file.read_text())["annotations"]
        assert annotation["iscrowd"] == iscrowd
        assert annotation["segmentation"] == segmentation

    def test_save__other_input_is_not_crowd(self, tmp_path: Path) -> None:
        # The input has one polygon and one binary mask, both without a crowd flag.
        output_file = tmp_path / "output.json"

        COCOInstanceSegmentationOutput(output_file=output_file).save(
            label_input=SimpleInstanceSegmentationInputWithBinaryMask()
        )

        annotations = json.loads(output_file.read_text())["annotations"]
        assert [type(annotation["segmentation"]) for annotation in annotations] == [
            list,
            dict,
        ]
        assert [annotation["iscrowd"] for annotation in annotations] == [0, 0]

    def test_save__id_and_area(self, tmp_path: Path) -> None:
        # The input has a 10x10 polygon and a binary mask with 2 foreground pixels.
        output_file = tmp_path / "output.json"

        COCOInstanceSegmentationOutput(output_file=output_file).save(
            label_input=SimpleInstanceSegmentationInputWithBinaryMask()
        )

        annotations = json.loads(output_file.read_text())["annotations"]
        assert [annotation["id"] for annotation in annotations] == [1, 2]
        assert [annotation["area"] for annotation in annotations] == [100.0, 2.0]

    def test_save__area_matches_coco_dataset(self, tmp_path: Path) -> None:
        # The fixture holds polygon annotations from the COCO dataset with their areas.
        input_file = _FIXTURES_DIR / "instance_segmentation/COCO/instances.json"
        output_file = tmp_path / "output.json"

        COCOInstanceSegmentationOutput(output_file=output_file).save(
            label_input=COCOInstanceSegmentationInput(input_file=input_file)
        )

        input_annotations = json.loads(input_file.read_text())["annotations"]
        output_annotations = json.loads(output_file.read_text())["annotations"]
        assert [annotation["area"] for annotation in output_annotations] == [
            pytest.approx(annotation["area"]) for annotation in input_annotations
        ]


def _create_coco_instance_segmentation_file(
    tmp_path: Path, segmentation: object, iscrowd: object
) -> Path:
    data = {
        "images": [{"id": 0, "file_name": "image.jpg", "width": 3, "height": 2}],
        "categories": [{"id": 1, "name": "dog"}],
        "annotations": [
            {
                "image_id": 0,
                "category_id": 1,
                "bbox": [1.0, 0.0, 2.0, 2.0],
                "iscrowd": iscrowd,
                "segmentation": segmentation,
            }
        ],
    }
    coco_file = tmp_path / "instances.json"
    coco_file.write_text(json.dumps(data))
    return coco_file
