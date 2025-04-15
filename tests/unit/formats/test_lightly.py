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

from ... import simple_object_detection_label_input


def _create_label_file(tmp_path: Path, filename: str) -> Path:
    """Create a dummy label file in the given directory."""
    annotation = json.dumps(
        {
            "file_name": filename,
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
    label_path = (tmp_path / "labels" / filename).with_suffix(".json")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(annotation)
    return label_path


def _create_schema_file(tmp_path: Path) -> Path:
    """Create a dummy schema file in the given directory."""
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
    return schema_path


class TestLightlyObjectDetectionInput:
    def test_get_labels(self, tmp_path: Path, mocker: MockerFixture) -> None:
        # Prepare inputs.
        _create_label_file(tmp_path=tmp_path, filename="image.jpg")
        _create_label_file(tmp_path=tmp_path, filename="subdir/image.jpg")
        _create_schema_file(tmp_path=tmp_path)

        # Mock the image file.
        (tmp_path / "images").mkdir()
        (tmp_path / "images/image.jpg").touch()
        (tmp_path / "images/subdir").mkdir(parents=True)
        (tmp_path / "images/subdir/image.jpg").touch()
        mock_img = mocker.MagicMock()
        mock_img.size = (100, 200)
        mock_context_manager = mocker.MagicMock()
        mock_context_manager.__enter__.return_value = mock_img
        mocker.patch("PIL.Image.open", return_value=mock_context_manager)

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
            ),
            ImageObjectDetection(
                image=Image(id=1, filename="subdir/image.jpg", width=100, height=200),
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
            ),
        ]

    def test_get_labels__raises_label_without_image(self, tmp_path: Path) -> None:
        # Prepare inputs.
        _create_label_file(tmp_path=tmp_path, filename="image.jpg")
        _create_schema_file(tmp_path=tmp_path)

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

    def test_get_labels__skip_label_without_image(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        # Prepare inputs.
        _create_label_file(tmp_path=tmp_path, filename="image.jpg")
        _create_schema_file(tmp_path=tmp_path)

        # Convert.
        label_input = LightlyObjectDetectionInput(
            input_folder=tmp_path / "labels",
            images_rel_path="../images",
            skip_labels_without_image=True,
        )
        assert list(label_input.get_labels()) == []


class TestLightlyObjectDetectionOutput:
    @pytest.mark.parametrize("filename", ["image.jpg", "subdir1/subdir2/image.jpg"])
    @pytest.mark.parametrize("with_confidence", [True, False])
    def test_save(self, filename: str, tmp_path: Path, with_confidence: bool) -> None:
        output_folder = tmp_path / "labels"
        LightlyObjectDetectionOutput(output_folder=output_folder).save(
            label_input=simple_object_detection_label_input.get_input(
                filename=filename, with_confidence=with_confidence
            )
        )
        assert output_folder.exists()
        assert output_folder.is_dir()

        expected_label_filepath = (output_folder / filename).with_suffix(".json")
        filepaths = set(list(output_folder.glob("**/*.json")))
        assert len(filepaths) == 2
        assert filepaths == set(
            [
                expected_label_filepath,
                output_folder / "schema.json",
            ]
        )

        contents = expected_label_filepath.read_text()
        expected = json.dumps(
            {
                "file_name": filename,
                "predictions": [
                    {
                        "category_id": 1,
                        "bbox": [10.0, 20.0, 20.0, 20.0],
                        "score": 0.4 if with_confidence else 0.0,
                    },
                    {
                        "category_id": 0,
                        "bbox": [50.0, 60.0, 20.0, 20.0],
                        "score": 0.8 if with_confidence else 0.0,
                    },
                ],
            },
            indent=2,
        )
        assert contents == expected
