from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.category import Category
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.video import Video


@dataclass(frozen=True)
class SingleInstanceSegmentationTrack:
    category: Category
    segmentations: list[MultiPolygon | BinaryMaskSegmentation | None]


@dataclass(frozen=True)
class VideoInstanceSegmentationTrack:
    """
    The base class for a video alongside with its object detection track annotations.
    A video consists of N frames and M objects. Each object is defined by N boxes - one for each frame.
    If an object is not present on a frame, the corresponding entry is set to None.
    """

    video: Video
    objects: list[SingleInstanceSegmentationTrack]

    def __post_init__(self) -> None:
        number_of_frames = self.video.number_of_frames

        for obj in self.objects:
            if len(obj.segmentations) != number_of_frames:
                raise ValueError(
                    "Length of instance segmentation track does not match the number of frames in the video."
                )


class InstanceSegmentationTrackInput(ABC):
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
    def get_labels(self) -> Iterable[VideoInstanceSegmentationTrack]:
        raise NotImplementedError()


class InstanceSegmentationTrackOutput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    def save(self, label_input: InstanceSegmentationTrackInput) -> None:
        raise NotImplementedError()
