from pathlib import Path

from pytest_mock import MockerFixture

from labelformat.formats.coco import (
    COCOInstanceSegmentationInput,
    COCOInstanceSegmentationOutput,
)
from labelformat.formats.yolov8 import (
    YOLOv8InstanceSegmentationInput,
    YOLOv8InstanceSegmentationOutput,
)

from ...simple_instance_segmentation_label_input import SimpleInstanceSegmentationInput
from .. import integration_utils


def test_coco_inverse(tmp_path: Path) -> None:
    start_label_input = SimpleInstanceSegmentationInput()
    COCOInstanceSegmentationOutput(output_file=tmp_path / "train.json").save(
        label_input=start_label_input
    )
    end_label_input = COCOInstanceSegmentationInput(input_file=tmp_path / "train.json")
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_yolov8_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = SimpleInstanceSegmentationInput()
    YOLOv8InstanceSegmentationOutput(
        output_file=tmp_path / "dataset.yaml",
        output_split="train",
    ).save(label_input=start_label_input)
    # For YOLOv8 we have to also provide the image files.
    _mock_input_images(mocker=mocker, folder=tmp_path / "images")
    end_label_input = YOLOv8InstanceSegmentationInput(
        input_file=tmp_path / "dataset.yaml",
        input_split="train",
    )

    # YOLOv8 merges a multipolygon into a single polygon, so we have to
    # compare them with a custom check.
    for image_label_0, image_label_1 in zip(
        start_label_input.get_labels(), end_label_input.get_labels()
    ):
        assert image_label_0.image == image_label_1.image
        assert len(image_label_0.objects) == len(image_label_1.objects)
        for object_0, object_1 in zip(image_label_0.objects, image_label_1.objects):
            assert object_0.category == object_1.category
            integration_utils.assert_multipolygons_almost_equal(
                object_0.segmentation, object_1.segmentation
            )


def _mock_input_images(mocker: MockerFixture, folder: Path) -> None:
    folder.mkdir()
    (folder / "image.jpg").touch()
    mocker.patch("PIL.Image.open", autospec=True).return_value.size = (100, 200)
