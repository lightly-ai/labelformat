import json
from pathlib import Path

import pytest

from labelformat.formats.activitynet import ActivityNetTemporalClassificationInput
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
                subset="validation",
                resolution="270x480",
                url="https://www.youtube.com/watch?v=v_test_video",
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

    def test_filters_by_split(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_database_json(tmp_path / "activity_net.json")

        validation = ActivityNetTemporalClassificationInput(
            input_file=input_file, input_split="validation"
        )
        assert [label.video_id for label in validation.get_labels()] == ["v_test_video"]

    def test_rejects_unknown_split(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_database_json(tmp_path / "activity_net.json")

        with pytest.raises(ParseError, match="Split 'training' not found"):
            ActivityNetTemporalClassificationInput(
                input_file=input_file, input_split="training"
            )

    def test_rejects_segment_exceeding_duration(self, tmp_path: Path) -> None:
        input_file = tmp_path / "invalid.json"
        input_file.write_text(
            json.dumps(
                {
                    "database": {
                        "v_test_video": {
                            "duration": 5.0,
                            "subset": "validation",
                            "resolution": "270x480",
                            "url": "https://www.youtube.com/watch?v=v_test_video",
                            "annotations": [
                                {"label": "Person walking", "segment": [1.0, 6.0]},
                            ],
                        }
                    }
                }
            )
        )

        with pytest.raises(ParseError, match="exceed the video duration"):
            ActivityNetTemporalClassificationInput(input_file=input_file)

    def test_rejects_missing_required_field(self, tmp_path: Path) -> None:
        input_file = tmp_path / "invalid.json"
        input_file.write_text(
            json.dumps(
                {
                    "database": {
                        "v_test_video": {
                            "duration": 82.75,
                            "resolution": "270x480",
                            "url": "https://www.youtube.com/watch?v=v_test_video",
                            "annotations": [
                                {"label": "Person walking", "segment": [0.58, 6.16]},
                            ],
                        }
                    }
                }
            )
        )

        with pytest.raises(ParseError, match="missing required field 'subset'"):
            ActivityNetTemporalClassificationInput(input_file=input_file)

    def test_optional_metadata_defaults_to_none(self, tmp_path: Path) -> None:
        input_file = tmp_path / "no_optional.json"
        input_file.write_text(
            json.dumps(
                {
                    "database": {
                        "v_test_video": {
                            "duration": 82.75,
                            "subset": "validation",
                            "annotations": [
                                {"label": "Person walking", "segment": [0.58, 6.16]},
                            ],
                        }
                    }
                }
            )
        )

        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)
        labels = list(label_input.get_labels())

        assert len(labels) == 1
        assert labels[0].duration_s == 82.75
        assert labels[0].subset == "validation"
        assert labels[0].resolution is None
        assert labels[0].url is None


class TestActivityNetTemporalClassificationResultsInput:
    def test_get_labels(self, tmp_path: Path) -> None:
        input_file = _write_activitynet_results_json(tmp_path / "results.json")
        label_input = ActivityNetTemporalClassificationInput(input_file=input_file)

        assert list(label_input.get_labels()) == [
            VideoTemporalClassification(
                video_id="v_test_video",
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


def _write_activitynet_database_json(input_file: Path) -> Path:
    data = {
        "database": {
            "v_test_video": {
                "duration": 82.75,
                "subset": "validation",
                "resolution": "270x480",
                "url": "https://www.youtube.com/watch?v=v_test_video",
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
