import json
from pathlib import Path

from labelformat.formats.youtubevis import (
    YouTubeVISInstanceSegmentationTrackInput,
    YouTubeVISInstanceSegmentationTrackOutput,
)

from ..integration_utils import INST_SEGMENTATION_FIXTURES_DIR


def test_youtubevis_instance_segmentation_import_export(tmp_path: Path) -> None:
    youtubevis_file = INST_SEGMENTATION_FIXTURES_DIR / "YouTubeVIS/sample.json"
    label_input = YouTubeVISInstanceSegmentationTrackInput(input_file=youtubevis_file)

    output_path = tmp_path / "sample_out.json"
    YouTubeVISInstanceSegmentationTrackOutput(output_file=output_path).save(
        label_input=label_input
    )

    output_json = json.loads(output_path.read_text())
    expected_json = json.loads(youtubevis_file.read_text())

    assert output_json == expected_json

