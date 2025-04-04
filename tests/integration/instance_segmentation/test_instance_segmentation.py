import json
from pathlib import Path

from labelformat.formats.coco import (
    COCOInstanceSegmentationInput,
    COCOInstanceSegmentationOutput,
)

from ..integration_utils import (
    INST_SEGMENTATION_FIXTURES_DIR,
    assert_almost_equal_recursive,
)


def test_coco_to_coco(tmp_path: Path) -> None:
    coco_file = INST_SEGMENTATION_FIXTURES_DIR / "COCO/instances_with_binary_mask.json"
    label_input = COCOInstanceSegmentationInput(input_file=coco_file)
    COCOInstanceSegmentationOutput(
        output_file=tmp_path / "instances_with_binary_mask.json"
    ).save(label_input=label_input)

    # Compare jsons.
    output_json = json.loads((tmp_path / "instances_with_binary_mask.json").read_text())
    expected_json = json.loads(
        (
            INST_SEGMENTATION_FIXTURES_DIR / "COCO/instances_with_binary_mask.json"
        ).read_text()
    )
    # Some fields are not converted:
    # - info
    # - licenses
    # - <category>.supercategory
    # - <image>.date_captured
    # - <image>.license
    # - <image>.flickr_url
    # - <image>.coco_url
    # - <annotation>.id
    # - <annotation>.area
    del expected_json["info"]
    del expected_json["licenses"]
    for category in expected_json["categories"]:
        del category["supercategory"]
    for image in expected_json["images"]:
        del image["date_captured"]
        del image["license"]
        del image["flickr_url"]
        del image["coco_url"]
    for annotation in expected_json["annotations"]:
        del annotation["id"]
        del annotation["area"]
    assert_almost_equal_recursive(output_json, expected_json)
