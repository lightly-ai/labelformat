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
        raise ValueError(
            "LabelformatObjectDetectionInput does not support CLI arguments"
        )

    def get_categories(self) -> Iterable[Category]:
        return self.categories

    def get_images(self) -> Iterable[Image]:
        return self.images


@dataclass
class LabelformatObjectDetectionInput(_CustomBaseInput, ObjectDetectionInput):

    labels: list[ImageObjectDetection]

    """Class for custom object detection input format. 
    
    It can be used standalone or for conversion to other formats.

    """

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        return self.labels
