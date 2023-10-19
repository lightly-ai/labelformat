from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, List

from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.multipolygon import MultiPolygon


@dataclass(frozen=True)
class SingleInstanceSegmentation:
    category: Category
    segmentation: MultiPolygon


@dataclass(frozen=True)
class ImageInstanceSegmentation:
    image: Image
    objects: List[SingleInstanceSegmentation]


class InstanceSegmentationInput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_images(self) -> Iterable[Image]:
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        raise NotImplementedError()


class InstanceSegmentationOutput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    def save(self, label_input: InstanceSegmentationInput) -> None:
        raise NotImplementedError()
