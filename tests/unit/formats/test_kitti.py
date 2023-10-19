from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

from pytest_mock import MockerFixture

from labelformat.formats.kitti import (
    KittiObjectDetectionInput,
    KittiObjectDetectionOutput,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)

from ...simple_object_detection_label_input import SimpleObjectDetectionInput


class TestKittiObjectDetectionInput:
    def test_get_labels(self, tmp_path: Path, mocker: MockerFixture) -> None:
        # Prepare inputs.
        annotation = (
            "dog -1 -1 -10 10.0 20.0 30.0 40.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
            "cat -1 -1 -10 50.0 60.0 70.0 80.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        label_path = tmp_path / "labels" / "image.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(annotation)

        # Mock the image file.
        (tmp_path / "images").mkdir()
        (tmp_path / "images/image.jpg").touch()
        mocker.patch("PIL.Image.open", autospec=True).return_value.size = (100, 200)

        # Convert.
        label_input = KittiObjectDetectionInput(
            input_folder=tmp_path / "labels", category_names="cat,dog,cow"
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


class TestKittiObjectDetectionOutput:
    def test_save(self, tmp_path: Path) -> None:
        output_folder = tmp_path / "labels"
        KittiObjectDetectionOutput(output_folder=output_folder).save(
            label_input=SimpleObjectDetectionInput()
        )
        assert output_folder.exists()
        assert output_folder.is_dir()

        filepaths = list(output_folder.glob("**/*"))
        assert len(filepaths) == 1
        path = filepaths[0]
        assert path == tmp_path / "labels" / "image.txt"

        contents = path.read_text()
        expected = (
            "dog -1 -1 -10 10.0 20.0 30.0 40.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
            "cat -1 -1 -10 50.0 60.0 70.0 80.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        assert contents == expected
