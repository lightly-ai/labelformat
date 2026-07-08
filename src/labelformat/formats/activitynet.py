from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from labelformat.model.category import Category
from labelformat.model.temporal_classification import (
    TemporalClassificationInput,
    TemporalClassificationOutput,
    TemporalEvent,
    VideoTemporalClassification,
)
from labelformat.types import JsonDict, ParseError


class _ActivityNetBaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to input ActivityNet JSON file",
        )

    def __init__(self, input_file: Path) -> None:
        with input_file.open(encoding="utf-8") as file:
            data = json.load(file)
        self._labels, self._categories = _parse_activitynet_data(data=data)

    def get_categories(self) -> Iterable[Category]:
        yield from self._categories

    def get_labels(self) -> Iterable[VideoTemporalClassification]:
        yield from self._labels


class ActivityNetTemporalClassificationInput(
    _ActivityNetBaseInput, TemporalClassificationInput
):
    """Import ActivityNet-style temporal classification annotations."""


class _ActivityNetBaseOutput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-file",
            type=Path,
            required=True,
            help="Path to output ActivityNet JSON file",
        )

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file


class ActivityNetTemporalClassificationResultsOutput(
    _ActivityNetBaseOutput, TemporalClassificationOutput
):
    """Export ActivityNet submission-style JSON with a top-level ``results`` key."""

    def save(self, label_input: TemporalClassificationInput) -> None:
        data: JsonDict = {
            "results": _get_output_results_dict(label_input.get_labels()),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


class ActivityNetTemporalClassificationDatabaseOutput(
    _ActivityNetBaseOutput, TemporalClassificationOutput
):
    """Export ActivityNet ground-truth-style JSON with a top-level ``database`` key."""

    def save(self, label_input: TemporalClassificationInput) -> None:
        data: JsonDict = {
            "database": _get_output_database_dict(label_input.get_labels()),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def _parse_activitynet_data(
    data: JsonDict,
) -> tuple[list[VideoTemporalClassification], list[Category]]:
    if "database" in data:
        return _parse_database(database=data["database"])
    if "results" in data:
        return _parse_results(results=data["results"])

    raise ParseError("ActivityNet JSON must contain a 'database' or 'results' key.")


def _parse_database(
    database: JsonDict,
) -> tuple[list[VideoTemporalClassification], list[Category]]:
    label_names: dict[str, None] = {}
    parsed_by_video: list[tuple[str, list[_ParsedEvent], float | None]] = []

    for video_id, video_entry in database.items():
        if not isinstance(video_entry, dict):
            raise ParseError(f"Invalid database entry for video '{video_id}'.")
        raw_annotations = video_entry.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise ParseError(f"Invalid annotations for video '{video_id}'.")
        duration = video_entry.get("duration")
        duration_s = float(duration) if duration is not None else None
        events = [_parse_event(annotation=annotation) for annotation in raw_annotations]
        label_names.update((event.label, None) for event in events)
        parsed_by_video.append((str(video_id), events, duration_s))

    categories = _categories_from_label_names(label_names=label_names)
    category_name_to_id = {category.name: category.id for category in categories}
    labels = [
        VideoTemporalClassification(
            video_id=video_id,
            events=_events_from_parsed(
                events=events, category_name_to_id=category_name_to_id
            ),
            duration_s=duration_s,
        )
        for video_id, events, duration_s in parsed_by_video
    ]
    return labels, categories


def _parse_results(
    results: JsonDict,
) -> tuple[list[VideoTemporalClassification], list[Category]]:
    label_names: dict[str, None] = {}
    parsed_by_video: list[tuple[str, list[_ParsedEvent]]] = []

    for video_id, raw_annotations in results.items():
        if not isinstance(raw_annotations, list):
            raise ParseError(f"Invalid results entry for video '{video_id}'.")
        events = [_parse_event(annotation=annotation) for annotation in raw_annotations]
        label_names.update((event.label, None) for event in events)
        parsed_by_video.append((str(video_id), events))

    categories = _categories_from_label_names(label_names=label_names)
    category_name_to_id = {category.name: category.id for category in categories}
    labels = [
        VideoTemporalClassification(
            video_id=video_id,
            events=_events_from_parsed(
                events=events, category_name_to_id=category_name_to_id
            ),
            duration_s=None,
        )
        for video_id, events in parsed_by_video
    ]
    return labels, categories


@dataclass(frozen=True)
class _ParsedEvent:
    label: str
    start_time_s: float
    end_time_s: float
    confidence: float | None


def _parse_event(annotation: JsonDict) -> _ParsedEvent:
    label = annotation.get("label")
    segment = annotation.get("segment")
    if not isinstance(label, str) or not label:
        raise ParseError("ActivityNet event must contain a non-empty 'label'.")
    if not isinstance(segment, list) or len(segment) != 2:
        raise ParseError(
            "ActivityNet event must contain 'segment' as [start_s, end_s]."
        )

    start_time_s = float(segment[0])
    end_time_s = float(segment[1])
    if start_time_s < 0 or start_time_s >= end_time_s:
        raise ParseError(
            f"Invalid segment [{start_time_s}, {end_time_s}] for label '{label}': "
            "start must be non-negative and less than end."
        )

    confidence = annotation.get("score")
    if confidence is not None:
        confidence = float(confidence)

    return _ParsedEvent(
        label=label,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        confidence=confidence,
    )


def _categories_from_label_names(label_names: Iterable[str]) -> list[Category]:
    return [
        Category(id=index, name=label_name)
        for index, label_name in enumerate(label_names, start=1)
    ]


def _events_from_parsed(
    events: list[_ParsedEvent],
    category_name_to_id: dict[str, int],
) -> list[TemporalEvent]:
    return [
        TemporalEvent(
            category=Category(
                id=category_name_to_id[event.label],
                name=event.label,
            ),
            start_time_s=event.start_time_s,
            end_time_s=event.end_time_s,
            confidence=event.confidence,
        )
        for event in events
    ]


def _get_output_results_dict(
    labels: Iterable[VideoTemporalClassification],
) -> JsonDict:
    results: JsonDict = {}
    for label in labels:
        annotations: list[JsonDict] = []
        for event in label.events:
            annotation: JsonDict = {
                "label": event.category.name,
                "segment": [event.start_time_s, event.end_time_s],
            }
            if event.confidence is not None:
                annotation["score"] = event.confidence
            annotations.append(annotation)
        results[label.video_id] = annotations
    return results


def _get_output_database_dict(
    labels: Iterable[VideoTemporalClassification],
) -> JsonDict:
    database: JsonDict = {}
    for label in labels:
        annotations: list[JsonDict] = []
        for event in label.events:
            annotation: JsonDict = {
                "label": event.category.name,
                "segment": [event.start_time_s, event.end_time_s],
            }
            if event.confidence is not None:
                annotation["score"] = event.confidence
            annotations.append(annotation)
        video_entry: JsonDict = {"annotations": annotations}
        if label.duration_s is not None:
            video_entry["duration"] = label.duration_s
        database[label.video_id] = video_entry
    return database
