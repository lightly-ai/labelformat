from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, List

from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.video import Video


@dataclass(frozen=True)
class SingleObjectDetectionTrack:
    category: Category
    boxes: list[BoundingBox | None]
    # TODO (Jonas, 01/2026): Add confidence


@dataclass(frozen=True)
class VideoObjectDetectionTrack:
    """
    The base class for a video alongside with its object detection track annotations.
    A video consists of N frames and M objects. Each object is defined by N boxes - one for each frame.
    If an object is not present on a frame, the corresponding entry is set to None.
    """

    video: Video
    objects: List[SingleObjectDetectionTrack]

    def __post_init__(self) -> None:
        number_of_frames = self.video.number_of_frames

        for obj in self.objects:
            if len(obj.boxes) != number_of_frames:
                raise ValueError(
                    "Length of object detection track does not match the number of frames in the video."
                )


class ObjectDetectionTrackInput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_videos(self) -> Iterable[Video]:
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> Iterable[VideoObjectDetectionTrack]:
        raise NotImplementedError()


class ObjectDetectionTrackOutput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    def save(self, label_input: ObjectDetectionTrackInput) -> None:
        raise NotImplementedError()
