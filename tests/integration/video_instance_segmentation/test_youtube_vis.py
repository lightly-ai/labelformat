import json
from pathlib import Path

import pytest
from labelformat.formats.youtube_vis import YouTubeVISInput, YouTubeVISOutput
from labelformat.model.category import Category
from labelformat.model.video import Video
from tests.integration.integration_utils import VIDEO_INSTANCE_SEGMENTATION_FIXTURES_DIR
from tests.integration.integration_utils import assert_almost_equal_recursive
REAL_DATA_FILE = VIDEO_INSTANCE_SEGMENTATION_FIXTURES_DIR / "OVIS" / "train" / "annotations_train.json"

def test_youtube_vis_input_with_real_data() -> None:
    label_input = YouTubeVISInput(input_file=REAL_DATA_FILE)

    categories = list(label_input.get_categories())
    # Add assertions based on the expected categories in your real data
    assert categories  # Ensure categories are not empty

    videos = list(label_input.get_videos())
    # Add assertions based on the expected videos in your real data
    assert videos  # Ensure videos are not empty

    labels = list(label_input.get_labels())
    # Add assertions based on the expected labels in your real data
    assert labels  # Ensure labels are not empty

def test_youtube_vis_output_with_real_data(tmp_path: Path) -> None:
    label_input = YouTubeVISInput(input_file=REAL_DATA_FILE)
    output_file = tmp_path / "output.json"
    label_output = YouTubeVISOutput(output_file=output_file)

    label_output.save(label_input=label_input)

    output_data = json.loads(output_file.read_text())
    assert "videos" in output_data
    assert "categories" in output_data
    assert "annotations" in output_data

    # Add assertions based on the expected output structure and content
    assert output_data["videos"]  # Ensure videos are not empty
    assert output_data["categories"]  # Ensure categories are not empty
    assert output_data["annotations"]  # Ensure annotations are not empty


def test_youtube_vis_to_youtube_vis(tmp_path: Path) -> None:
    
    label_input = YouTubeVISInput(input_file=REAL_DATA_FILE)
    YouTubeVISOutput(output_file=tmp_path / "annotations_train.json").save(
        label_input=label_input
    )

    # Compare jsons.
    output_json = json.loads((tmp_path / "annotations_train.json").read_text())
    expected_json = json.loads(
        REAL_DATA_FILE.read_text()
    )
    
    # Remove fields that are not converted or are expected to differ
    if "info" in expected_json:
        del expected_json["info"]

    if "licenses" in expected_json:
        del expected_json["licenses"]

    for category in expected_json["categories"]:
        del category["supercategory"]

    for video in expected_json["videos"]:
        del video["license"]

    for annotation in expected_json["annotations"]:
        del annotation["areas"]
        del annotation["bboxes"]
        del annotation["length"]
        del annotation["occlusion"]

    for annotation in output_json["annotations"]:
        del annotation["occlusion"]


    assert_almost_equal_recursive(output_json, expected_json)