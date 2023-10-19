import json
from pathlib import Path

import pytest

from labelformat.formats.coco import COCOObjectDetectionInput, COCOObjectDetectionOutput
from labelformat.formats.kitti import (
    KittiObjectDetectionInput,
    KittiObjectDetectionOutput,
)
from labelformat.formats.lightly import (
    LightlyObjectDetectionInput,
    LightlyObjectDetectionOutput,
)

from ..integration_utils import COMMA_JOINED_CATEGORY_NAMES, OBJ_DETECTION_FIXTURES_DIR


def test_coco_to_coco(tmp_path: Path) -> None:
    coco_file = OBJ_DETECTION_FIXTURES_DIR / "COCO/train.json"
    label_input = COCOObjectDetectionInput(input_file=coco_file)
    COCOObjectDetectionOutput(output_file=tmp_path / "train.json").save(
        label_input=label_input
    )

    # Compare jsons.
    output_json = json.loads((tmp_path / "train.json").read_text())
    expected_json = json.loads(
        (OBJ_DETECTION_FIXTURES_DIR / "COCO/train.json").read_text()
    )
    # Some fields are not converted:
    # - info
    # - licenses
    # - <category>.supercategory
    # - <image>.date_captured
    # - <annotation>.id
    # - <annotation>.area
    # - <annotation>.iscrowd
    del expected_json["info"]
    del expected_json["licenses"]
    for category in expected_json["categories"]:
        del category["supercategory"]
    for image in expected_json["images"]:
        del image["date_captured"]
    for annotation in expected_json["annotations"]:
        del annotation["id"]
        del annotation["area"]
        del annotation["iscrowd"]
    assert output_json == expected_json


def test_kitti_to_kitti(tmp_path: Path) -> None:
    input_folder = OBJ_DETECTION_FIXTURES_DIR / "KITTI/labels"
    label_input = KittiObjectDetectionInput(
        input_folder=input_folder,
        category_names=COMMA_JOINED_CATEGORY_NAMES,
        images_rel_path="../images/a-difficult subfolder",
    )
    output_folder = tmp_path / "labels"
    KittiObjectDetectionOutput(output_folder=output_folder).save(
        label_input=label_input
    )

    # Compare kitti files.
    for file1 in input_folder.rglob("*"):
        if file1.is_dir():
            continue
        file2 = output_folder / file1.relative_to(input_folder)

        contents1 = file1.read_text()
        contents2 = file2.read_text()
        assert contents1 == contents2


def test_lightly_to_lightly(tmp_path: Path) -> None:
    input_folder = OBJ_DETECTION_FIXTURES_DIR / "lightly/detection-task-name"
    label_input = LightlyObjectDetectionInput(
        input_folder=input_folder,
        images_rel_path="../images",
    )
    output_folder = tmp_path / "detection-task-name"
    LightlyObjectDetectionOutput(output_folder=output_folder).save(
        label_input=label_input
    )

    # Compare Json files.
    for file1 in input_folder.rglob("*.json"):
        if file1.is_dir():
            continue
        file2 = output_folder / file1.relative_to(input_folder)

        contents1 = json.loads(file1.read_text())
        contents2 = json.loads(file2.read_text())
        assert contents1 == contents2
