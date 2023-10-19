from pathlib import Path

import yaml

from labelformat.formats.coco import COCOInstanceSegmentationInput
from labelformat.formats.yolov8 import (
    YOLOv8InstanceSegmentationInput,
    YOLOv8InstanceSegmentationOutput,
)
from labelformat.model.instance_segmentation import InstanceSegmentationInput
from labelformat.model.multipolygon import MultiPolygon

from .. import integration_utils
from ..integration_utils import INST_SEGMENTATION_FIXTURES_DIR


def test_yolov8_to_yolov8(tmp_path: Path) -> None:
    label_input = YOLOv8InstanceSegmentationInput(
        input_file=INST_SEGMENTATION_FIXTURES_DIR / "YOLOv8/dataset.yaml",
        input_split="train",
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def test_coco_to_yolov8(tmp_path: Path) -> None:
    label_input = COCOInstanceSegmentationInput(
        input_file=INST_SEGMENTATION_FIXTURES_DIR / "COCO/instances.json"
    )
    _convert_and_test(label_input=label_input, tmp_path=tmp_path)


def _convert_and_test(label_input: InstanceSegmentationInput, tmp_path: Path) -> None:
    YOLOv8InstanceSegmentationOutput(
        output_file=tmp_path / "dataset.yaml",
        output_split="train",
    ).save(label_input=label_input)

    # Compare yaml files.
    output_yaml = yaml.safe_load((tmp_path / "dataset.yaml").read_text())
    expected_yaml = yaml.safe_load(
        (INST_SEGMENTATION_FIXTURES_DIR / "YOLOv8/dataset.yaml").read_text()
    )
    # TODO: Add split_subfolder to YOLOv8 output parameters.
    del output_yaml["train"]
    del expected_yaml["train"]
    assert output_yaml == expected_yaml

    # Compare label files.
    _assert_yolov8_labels_equal(
        dir1=INST_SEGMENTATION_FIXTURES_DIR / "YOLOv8/labels",
        dir2=tmp_path / "labels",
    )


def _assert_yolov8_labels_equal(
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
            polygon1 = [
                (float(x), float(y)) for x, y in zip(parts1[1::2], parts1[2::2])
            ]
            polygon2 = [
                (float(x), float(y)) for x, y in zip(parts2[1::2], parts2[2::2])
            ]
            integration_utils.assert_multipolygons_almost_equal(
                MultiPolygon(polygons=[polygon1]),
                MultiPolygon(polygons=[polygon2]),
            )
