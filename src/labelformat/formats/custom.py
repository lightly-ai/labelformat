from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable
from dataclasses import dataclass

from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
)
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
)


@dataclass
class _CustomBaseInput:
    categories: list[Category]
    images: list[Image]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise ValueError("CustomObjectDetectionInput does not support CLI arguments")

    def get_categories(self) -> Iterable[Category]:
        return self.categories

    def get_images(self) -> Iterable[Image]:
        return self.images


@dataclass
class CustomObjectDetectionInput(_CustomBaseInput, ObjectDetectionInput):

    labels: list[ImageObjectDetection]

    """Class for custom object detection input format. 
    
    It can be used standalone or for conversion to other formats.

    """

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        return self.labels


@dataclass
class CustomInstanceSegmentationInput(_CustomBaseInput, InstanceSegmentationInput):

    labels: list[ImageInstanceSegmentation]

    """Class for custom instance segmentation input format.

    It can be used standalone or for conversion to other formats.

    Creation example: 
    ```python
    TODO(Malte, 04/2025): Add example
    ```

    """

    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        return self.labels
