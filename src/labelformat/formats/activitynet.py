from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from labelformat.model.category import Category
from labelformat.model.temporal_classification import (
    TemporalClassificationInput,
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


def _parse_activitynet_data(
    data: JsonDict,
) -> tuple[list[VideoTemporalClassification], list[Category]]:
    if "database" in data:
        entries = data["database"]
        is_database = True
    elif "results" in data:
        entries = data["results"]
        is_database = False
    else:
        raise ParseError(
            "ActivityNet JSON must contain a 'database' or 'results' key."
        )

    label_names: dict[str, None] = {}
    parsed_by_video: list[tuple[str, list[_ParsedEvent]]] = []
    for video_id, video_entry in entries.items():
        raw_annotations = _extract_annotations(
            video_id=video_id, video_entry=video_entry, is_database=is_database
        )
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
        )
        for video_id, events in parsed_by_video
    ]
    return labels, categories


def _extract_annotations(
    video_id: str,
    video_entry: object,
    is_database: bool,
) -> list[JsonDict]:
    """Extract the raw annotation list for one video.

    In the ``database`` format each entry is a dict with an ``annotations`` list;
    in the ``results`` format the entry is the list itself.
    """
    if is_database:
        if not isinstance(video_entry, dict):
            raise ParseError(f"Invalid database entry for video '{video_id}'.")
        raw_annotations = video_entry.get("annotations", [])
    else:
        raw_annotations = video_entry
    if not isinstance(raw_annotations, list):
        raise ParseError(f"Invalid annotations for video '{video_id}'.")
    return raw_annotations


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
