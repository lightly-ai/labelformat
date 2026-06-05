from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.multipolygon import MultiPolygon


@dataclass(frozen=True)
class SingleInstanceSegmentation:
    category: Category
    segmentation: MultiPolygon | BinaryMaskSegmentation
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(
                f"Confidence must be between 0 and 1, but got: {self.confidence}"
            )


@dataclass(frozen=True)
class ImageInstanceSegmentation:
    image: Image
    objects: list[SingleInstanceSegmentation]


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
