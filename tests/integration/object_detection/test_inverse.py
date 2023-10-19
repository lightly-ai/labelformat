from pathlib import Path

from pytest_mock import MockerFixture

from labelformat.formats.coco import COCOObjectDetectionInput, COCOObjectDetectionOutput
from labelformat.formats.kitti import (
    KittiObjectDetectionInput,
    KittiObjectDetectionOutput,
)
from labelformat.formats.lightly import (
    LightlyObjectDetectionInput,
    LightlyObjectDetectionOutput,
)
from labelformat.formats.pascalvoc import (
    PascalVOCObjectDetectionInput,
    PascalVOCObjectDetectionOutput,
)
from labelformat.formats.yolov8 import (
    YOLOv8ObjectDetectionInput,
    YOLOv8ObjectDetectionOutput,
)

from ...simple_object_detection_label_input import SimpleObjectDetectionInput


def test_coco_inverse(tmp_path: Path) -> None:
    start_label_input = SimpleObjectDetectionInput()
    COCOObjectDetectionOutput(output_file=tmp_path / "train.json").save(
        label_input=start_label_input
    )
    end_label_input = COCOObjectDetectionInput(input_file=tmp_path / "train.json")
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_yolov8_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = SimpleObjectDetectionInput()
    YOLOv8ObjectDetectionOutput(
        output_file=tmp_path / "data.yaml",
        output_split="train",
    ).save(label_input=start_label_input)

    # For YOLOv8 we have to also provide the image files.
    _mock_input_images(mocker=mocker, folder=tmp_path / "images")

    end_label_input = YOLOv8ObjectDetectionInput(
        input_file=tmp_path / "data.yaml",
        input_split="train",
    )
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_pascalvoc_inverse(tmp_path: Path) -> None:
    start_label_input = SimpleObjectDetectionInput()
    PascalVOCObjectDetectionOutput(output_folder=tmp_path).save(
        label_input=start_label_input
    )
    end_label_input = PascalVOCObjectDetectionInput(
        input_folder=tmp_path,
        category_names="cat,dog,cow",
    )
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_kitti_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = SimpleObjectDetectionInput()
    KittiObjectDetectionOutput(output_folder=tmp_path / "labels").save(
        label_input=start_label_input
    )

    # For KITTI we have to also provide the image files.
    _mock_input_images(mocker=mocker, folder=tmp_path / "images")

    end_label_input = KittiObjectDetectionInput(
        input_folder=tmp_path / "labels",
        category_names="cat,dog,cow",
    )
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_lightly_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = SimpleObjectDetectionInput()
    LightlyObjectDetectionOutput(output_folder=tmp_path / "task").save(
        label_input=start_label_input
    )

    # For Lightly we have to also provide the image files.
    _mock_input_images(mocker=mocker, folder=tmp_path / "images")

    end_label_input = LightlyObjectDetectionInput(
        input_folder=tmp_path / "task",
    )
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def _mock_input_images(mocker: MockerFixture, folder: Path) -> None:
    folder.mkdir()
    (folder / "image.jpg").touch()
    mocker.patch("PIL.Image.open", autospec=True).return_value.size = (100, 200)
