from pathlib import Path

import pytest
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
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)

from ... import simple_object_detection_label_input


def test_coco_inverse(tmp_path: Path) -> None:
    start_label_input = simple_object_detection_label_input.get_input()
    COCOObjectDetectionOutput(output_file=tmp_path / "train.json").save(
        label_input=start_label_input
    )
    end_label_input = COCOObjectDetectionInput(input_file=tmp_path / "train.json")
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_yolov8_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = simple_object_detection_label_input.get_input()
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
    start_label_input = simple_object_detection_label_input.get_input()
    PascalVOCObjectDetectionOutput(output_folder=tmp_path).save(
        label_input=start_label_input
    )
    end_label_input = PascalVOCObjectDetectionInput(
        input_folder=tmp_path,
        category_names="cat,dog,cow",
    )
    assert list(start_label_input.get_labels()) == list(end_label_input.get_labels())


def test_kitti_inverse(tmp_path: Path, mocker: MockerFixture) -> None:
    start_label_input = simple_object_detection_label_input.get_input()
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


@pytest.mark.parametrize("with_confidence", [True, False])
def test_lightly_inverse(
    tmp_path: Path, mocker: MockerFixture, with_confidence: bool
) -> None:
    start_label_input = simple_object_detection_label_input.get_input(
        with_confidence=with_confidence
    )
    LightlyObjectDetectionOutput(output_folder=tmp_path / "task").save(
        label_input=start_label_input
    )

    # For Lightly we have to also provide the image files.
    _mock_input_images(mocker=mocker, folder=tmp_path / "images")

    end_label_input = LightlyObjectDetectionInput(
        input_folder=tmp_path / "task",
    )

    if with_confidence:
        expected_labels = list(start_label_input.get_labels())
    else:
        # If confidence is not set, it is set to 0.0 in the output.
        # We need to set it to None in the expected labels.
        expected_labels = []
        for label in start_label_input.get_labels():
            expected_objects = []
            for obj in label.objects:
                if obj.confidence is None:
                    obj = SingleObjectDetection(
                        category=obj.category,
                        box=obj.box,
                        confidence=0.0,
                    )
                expected_objects.append(obj)
            expected_labels.append(
                ImageObjectDetection(
                    image=label.image,
                    objects=expected_objects,
                )
            )
    assert list(end_label_input.get_labels()) == expected_labels


def _mock_input_images(mocker: MockerFixture, folder: Path) -> None:
    folder.mkdir()
    (folder / "image.jpg").touch()
    mock_img = mocker.MagicMock()
    mock_img.size = (100, 200)
    mock_context_manager = mocker.MagicMock()
    mock_context_manager.__enter__.return_value = mock_img
    mocker.patch("PIL.Image.open", return_value=mock_context_manager)
