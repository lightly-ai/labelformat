import json
from pathlib import Path

import pytest

from labelformat.formats.activitynet import (
    ActivityNetTemporalClassificationDatabaseOutput,
    ActivityNetTemporalClassificationInput,
    ActivityNetTemporalClassificationResultsOutput,
)
from labelformat.model.category import Category
from labelformat.model.temporal_classification import (
    TemporalEvent,
    VideoTemporalClassification,
)
from labelformat.types import ParseError


class TestActivityNetTemporalClassificationDatabaseInput:
    def test_get_categories(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_database_json(tmp_path / "activity_net.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        assert list(label_input.get_categories()) == [
            Category(id=1, name="Person walking"),
            Category(id=2, name="Rock climbing"),
        ]

    def test_get_labels(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_database_json(tmp_path / "activity_net.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        assert list(label_input.get_labels()) == [
            VideoTemporalClassification(
                video_id="v_test_video",
                duration_s=82.75,
                events=[
                    TemporalEvent(
                        category=Category(id=1, name="Person walking"),
                        start_time_s=0.58,
                        end_time_s=6.16,
                    ),
                    TemporalEvent(
                        category=Category(id=2, name="Rock climbing"),
                        start_time_s=10.0,
                        end_time_s=20.0,
                        confidence=0.9,
                    ),
                ],
            )
        ]


class TestActivityNetTemporalClassificationResultsInput:
    def test_get_labels(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_results_json(tmp_path / "results.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        assert list(label_input.get_labels()) == [
            VideoTemporalClassification(
                video_id="v_test_video",
                duration_s=None,
                events=[
                    TemporalEvent(
                        category=Category(id=1, name="Person walking"),
                        start_time_s=0.58,
                        end_time_s=6.16,
                        confidence=0.75,
                    )
                ],
            )
        ]

    def test_rejects_invalid_top_level_key(self, tmp_path: Path) -> None:
        input_file = tmp_path / "invalid.json"
        input_file.write_text(json.dumps({"videos": {}}))

        with pytest.raises(ParseError, match="database' or 'results'"):
            ActivityNetTemporalClassificationInput(input_file=input_file)

    def test_rejects_invalid_segment(self, tmp_path: Path) -> None:
        input_file = tmp_path / "invalid.json"
        input_file.write_text(
            json.dumps(
                {
                    "results": {
                        "v_test_video": [
                            {"label": "Person walking", "segment": [1.0]},
                        ]
                    }
                }
            )
        )

        with pytest.raises(ParseError, match="segment"):
            ActivityNetTemporalClassificationInput(input_file=input_file)


class TestActivityNetTemporalClassificationExportImport:
    def test_database_import_export(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_database_json(tmp_path / "activity_net.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        output_path = tmp_path / "activity_net_out.json"
        ActivityNetTemporalClassificationDatabaseOutput(output_file=output_path).save(
            label_input=label_input
        )

        output_json = json.loads(output_path.read_text())
        expected_json = json.loads(input_file.read_text())
        assert output_json == expected_json

    def test_results_import_export(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_results_json(tmp_path / "results.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        output_path = tmp_path / "results_out.json"
        ActivityNetTemporalClassificationResultsOutput(output_file=output_path).save(
            label_input=label_input
        )

        output_json = json.loads(output_path.read_text())
        expected_json = json.loads(input_file.read_text())
        assert output_json == expected_json


def _write_activitynet_database_json(input_file: Path) -> Path:
    data = {
        "database": {
            "v_test_video": {
                "duration": 82.75,
                "annotations": [
                    {
                        "label": "Person walking",
                        "segment": [0.58, 6.16],
                    },
                    {
                        "label": "Rock climbing",
                        "segment": [10.0, 20.0],
                        "score": 0.9,
                    },
                ],
            }
        }
    }
    input_file.write_text(json.dumps(data))
    return input_file


def _write_activitynet_results_json(input_file: Path) -> Path:
    data = {
        "results": {
            "v_test_video": [
                {
                    "label": "Person walking",
                    "segment": [0.58, 6.16],
                    "score": 0.75,
                }
            ]
        }
    }
    input_file.write_text(json.dumps(data))
    return input_file
