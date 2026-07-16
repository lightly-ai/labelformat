from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable

from labelformat.model.category import Category


@dataclass(frozen=True)
class TemporalEvent:
    """A single temporal classification event on a video."""

    category: Category
    start_time_s: float
    end_time_s: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_time_s < 0 or self.start_time_s >= self.end_time_s:
            raise ValueError(
                f"Invalid segment [{self.start_time_s}, {self.end_time_s}] "
                f"for label '{self.category.name}': start must be non-negative and less than end."
            )


@dataclass(frozen=True)
class VideoTemporalClassification:
    """All temporal classification events for one video."""

    video_id: str
    events: list[TemporalEvent]
    duration_s: float | None = None
    subset: str | None = None
    resolution: str | None = None
    url: str | None = None


class TemporalClassificationInput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> Iterable[VideoTemporalClassification]:
        raise NotImplementedError()
