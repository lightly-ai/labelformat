from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, List

from labelformat.model.category import Category
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.video import Video


@dataclass(frozen=True)
class SingleVideoInstanceSegmentation:
    category: Category
    segmentation: List[MultiPolygon]


@dataclass(frozen=True)
class VideoInstanceSegmentation:
    video: Video
    objects: List[SingleVideoInstanceSegmentation]


class VideoInstanceSegmentationInput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_videos(self) -> Iterable[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> Iterable[VideoInstanceSegmentation]:
        raise NotImplementedError()


class VideoInstanceSegmentationOutput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    def save(self, label_input: VideoInstanceSegmentationInput) -> None:
        raise NotImplementedError()
