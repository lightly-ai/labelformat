import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from labelformat.errors import LabelWithoutImageError
from labelformat.formats.lightly import (
    LightlyObjectDetectionInput,
    LightlyObjectDetectionOutput,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)

from ...simple_object_detection_label_input import SimpleObjectDetectionInput


class TestLightlyObjectDetectionInput:
    def test_get_labels(self, tmp_path: Path, mocker: MockerFixture) -> None:
        # Prepare inputs.
        annotation = json.dumps(
            {
                "file_name": "image.jpg",
                "predictions": [
                    {
                        "category_id": 1,
                        "bbox": [10.0, 20.0, 20.0, 20.0],
                    },
                    {
                        "category_id": 0,
                        "bbox": [50.0, 60.0, 20.0, 20.0],
                    },
                ],
            }
        )
        label_path = tmp_path / "labels" / "image.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(annotation)

        schema = json.dumps(
            {
                "task_type": "object-detection",
                "categories": [
                    {"name": "cat", "id": 0},
                    {"name": "dog", "id": 1},
                    {"name": "cow", "id": 2},
                ],
            }
        )
        schema_path = tmp_path / "labels" / "schema.json"
        schema_path.write_text(schema)

        # Mock the image file.
        (tmp_path / "images").mkdir()
        (tmp_path / "images/image.jpg").touch()
        mocker.patch("PIL.Image.open", autospec=True).return_value.size = (100, 200)

        # Convert.
        label_input = LightlyObjectDetectionInput(
            input_folder=tmp_path / "labels",
            images_rel_path="../images",
        )
        labels = list(label_input.get_labels())
        assert labels == [
            ImageObjectDetection(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleObjectDetection(
                        category=Category(id=1, name="dog"),
                        box=BoundingBox(
                            xmin=10.0,
                            ymin=20.0,
                            xmax=30.0,
                            ymax=40.0,
                        ),
                    ),
                    SingleObjectDetection(
                        category=Category(id=0, name="cat"),
                        box=BoundingBox(
                            xmin=50.0,
                            ymin=60.0,
                            xmax=70.0,
                            ymax=80.0,
                        ),
                    ),
                ],
            )
        ]

    def test_get_labels__raises_label_without_image(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # Prepare inputs.
        annotation = json.dumps(
            {
                "file_name": "image.jpg",
                "predictions": [
                    {
                        "category_id": 0,
                        "bbox": [10.0, 20.0, 20.0, 20.0],
                    },
                    {
                        "category_id": 1,
                        "bbox": [50.0, 60.0, 20.0, 20.0],
                    },
                ],
            }
        )
        label_path = tmp_path / "labels" / "image.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(annotation)

        schema = json.dumps(
            {
                "task_type": "object-detection",
                "categories": [
                    {"name": "cat", "id": 0},
                    {"name": "dog", "id": 1},
                    {"name": "cow", "id": 2},
                ],
            }
        )
        schema_path = tmp_path / "labels" / "schema.json"
        schema_path.write_text(schema)

        # Try to convert.
        label_input = LightlyObjectDetectionInput(
            input_folder=tmp_path / "labels",
            images_rel_path="../images",
        )
        with pytest.raises(
            LabelWithoutImageError,
            match="Label 'image.json' does not have a corresponding image.",
        ):
            list(label_input.get_labels())


class TestLightlyObjectDetectionOutput:
    def test_save(self, tmp_path: Path) -> None:
        output_folder = tmp_path / "labels"
        LightlyObjectDetectionOutput(output_folder=output_folder).save(
            label_input=SimpleObjectDetectionInput()
        )
        assert output_folder.exists()
        assert output_folder.is_dir()

        filepaths = sorted(list(output_folder.glob("**/*")))
        assert filepaths == [
            tmp_path / "labels" / "image.json",
            tmp_path / "labels" / "schema.json",
        ]

        contents = (tmp_path / "labels" / "image.json").read_text()
        expected = json.dumps(
            {
                "file_name": "image.jpg",
                "predictions": [
                    {
                        "category_id": 1,
                        "bbox": [10.0, 20.0, 20.0, 20.0],
                        "score": 0.0,  # default
                    },
                    {
                        "category_id": 0,
                        "bbox": [50.0, 60.0, 20.0, 20.0],
                        "score": 0.0,  # default
                    },
                ],
            },
            indent=2,
        )
        assert contents == expected
