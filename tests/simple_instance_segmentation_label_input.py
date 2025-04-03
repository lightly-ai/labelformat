from argparse import ArgumentParser
from typing import Iterable

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon


class _BaseSimpleInstanceSegmentationInput(InstanceSegmentationInput):
    def get_categories(self) -> Iterable[Category]:
        return [
            Category(id=0, name="cat"),
            Category(id=1, name="dog"),
            Category(id=2, name="cow"),
        ]

    def get_images(self) -> Iterable[Image]:
        return [
            Image(id=0, filename="image.jpg", width=100, height=200),
        ]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        # Default implementation (no arguments)
        pass


class SimpleInstanceSegmentationInput(_BaseSimpleInstanceSegmentationInput):
    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        return [
            ImageInstanceSegmentation(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleInstanceSegmentation(
                        category=Category(id=1, name="dog"),
                        segmentation=MultiPolygon(
                            polygons=[
                                [
                                    (10.0, 10.0),
                                    (10.0, 20.0),
                                    (20.0, 20.0),
                                    (20.0, 10.0),
                                ],
                            ],
                        ),
                    ),
                    SingleInstanceSegmentation(
                        category=Category(id=0, name="cat"),
                        segmentation=MultiPolygon(
                            polygons=[
                                [
                                    (30.0, 30.0),
                                    (40.0, 40.0),
                                    (40.0, 30.0),
                                ],
                                [
                                    (50.0, 50.0),
                                    (60.0, 60.0),
                                    (60.0, 50.0),
                                ],
                            ],
                        ),
                    ),
                ],
            )
        ]


class SimpleInstanceSegmentationInputWithBinaryMask(
    _BaseSimpleInstanceSegmentationInput
):
    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        return [
            ImageInstanceSegmentation(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleInstanceSegmentation(
                        category=Category(id=1, name="dog"),
                        segmentation=MultiPolygon(
                            polygons=[
                                [
                                    (10.0, 10.0),
                                    (10.0, 20.0),
                                    (20.0, 20.0),
                                    (20.0, 10.0),
                                ],
                            ],
                        ),
                    ),
                    SingleInstanceSegmentation(
                        category=Category(id=0, name="cat"),
                        segmentation=BinaryMaskSegmentation(
                            _rle_row_wise=[1, 2, 3],
                            width=3,
                            height=2,
                            bounding_box=BoundingBox(
                                0.0,
                                0.0,
                                3.0,
                                2.0,
                            ),
                        ),
                    ),
                ],
            )
        ]
