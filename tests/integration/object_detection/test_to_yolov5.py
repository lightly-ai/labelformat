from pathlib import Path

import pytest
import yaml

from labelformat.formats.coco import COCOObjectDetectionInput
from labelformat.formats.kitti import KittiObjectDetectionInput
from labelformat.formats.labelbox import LabelboxObjectDetectionInput
from labelformat.formats.lightly import LightlyObjectDetectionInput
from labelformat.formats.pascalvoc import PascalVOCObjectDetectionInput
from labelformat.formats.yolov5 import (
    YOLOv5ObjectDetectionInput,
    YOLOv5ObjectDetectionOutput,
)
from labelformat.model.object_detection import ObjectDetectionInput

from ..integration_utils import COMMA_JOINED_CATEGORY_NAMES, OBJ_DETECTION_FIXTURES_DIR


def test_yolov5_to_yolov5(tmp_path: Path) -> None:
    label_input = YOLOv5ObjectDetectionInput(
        input_file=OBJ_DETECTION_FIXTURES_DIR / "YOLOv8/example.yaml",
        input_split="train",
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_coco_to_yolov5(tmp_path: Path) -> None:
    label_input = COCOObjectDetectionInput(
        input_file=OBJ_DETECTION_FIXTURES_DIR / "COCO/train.json"
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_pascalvoc_to_yolov5(tmp_path: Path) -> None:
    label_input = PascalVOCObjectDetectionInput(
        input_folder=OBJ_DETECTION_FIXTURES_DIR / "PascalVOC",
        category_names=COMMA_JOINED_CATEGORY_NAMES,
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_kitty_to_yolov5(tmp_path: Path) -> None:
    label_input = KittiObjectDetectionInput(
        input_folder=OBJ_DETECTION_FIXTURES_DIR / "KITTI/labels",
        category_names=COMMA_JOINED_CATEGORY_NAMES,
        images_rel_path="../images/a-difficult subfolder",
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_lightly_to_yolov5(tmp_path: Path) -> None:
    label_input = LightlyObjectDetectionInput(
        input_folder=OBJ_DETECTION_FIXTURES_DIR / "lightly/detection-task-name",
        images_rel_path="../images",
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_labelbox_to_yolov5(tmp_path: Path) -> None:
    label_input = LabelboxObjectDetectionInput(
        input_file=OBJ_DETECTION_FIXTURES_DIR / "Labelbox/export-result.ndjson",
        category_names=COMMA_JOINED_CATEGORY_NAMES,
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def _convert_and_test(label_input: ObjectDetectionInput, tmp_path: Path) -> None:
    YOLOv5ObjectDetectionOutput(
        output_file=tmp_path / "data.yaml",
        output_split="train",
    ).save(label_input=label_input)

    # Compare yaml files.
    output_yaml = yaml.safe_load((tmp_path / "data.yaml").read_text())
    expected_yaml = yaml.safe_load(
        (OBJ_DETECTION_FIXTURES_DIR / "YOLOv8/example.yaml").read_text()
    )
    # TODO: Add split_subfolder to YOLOv8 output parameters.
    del output_yaml["train"]
    del expected_yaml["train"]
    assert output_yaml == expected_yaml

    # Compare label files.
    _assert_yolov5_labels_equal(
        dir1=OBJ_DETECTION_FIXTURES_DIR / "YOLOv8/labels/a-difficult subfolder",
        dir2=tmp_path / "labels",
    )


def _assert_yolov5_labels_equal(
    dir1: Path,
    dir2: Path,
) -> None:
    assert dir1.is_dir()
    assert dir2.is_dir()
    for file1 in dir1.rglob("*"):
        if file1.is_dir():
            continue
        file2 = dir2 / file1.relative_to(dir1)
        for line1, line2 in zip(file1.open(), file2.open()):
            parts1 = line1.split()
            parts2 = line2.split()
            assert parts1[0] == parts2[0], "labels do not match"
            assert pytest.approx(float(parts1[1]), rel=1e-1) == float(parts2[1])
            assert pytest.approx(float(parts1[2]), rel=1e-1) == float(parts2[2])
            assert pytest.approx(float(parts1[3]), rel=1e-1) == float(parts2[3])
            assert pytest.approx(float(parts1[4]), rel=1e-1) == float(parts2[4])
