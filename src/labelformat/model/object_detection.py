from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, List

from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image


@dataclass(frozen=True)
class SingleObjectDetection:
    category: Category
    box: BoundingBox


@dataclass(frozen=True)
class ImageObjectDetection:
    image: Image
    objects: List[SingleObjectDetection]


class ObjectDetectionInput(ABC):
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
    def get_labels(self) -> Iterable[ImageObjectDetection]:
        raise NotImplementedError()


class ObjectDetectionOutput(ABC):
    @staticmethod
    @abstractmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()

    def save(self, label_input: ObjectDetectionInput) -> None:
        raise NotImplementedError()
