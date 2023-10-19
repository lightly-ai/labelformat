from argparse import ArgumentParser
from typing import Iterable

from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)


class SimpleObjectDetectionInput(ObjectDetectionInput):
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

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        return [
            ImageObjectDetection(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleObjectDetection(
                        category=Category(id=1, name="dog"),
                        box=BoundingBox(
                            xmin=10.0,
                            ymin=20.0,
                            xmax=30.0,
                            ymax=40.0,
                        ),
                    ),
                    SingleObjectDetection(
                        category=Category(id=0, name="cat"),
                        box=BoundingBox(
                            xmin=50.0,
                            ymin=60.0,
                            xmax=70.0,
                            ymax=80.0,
                        ),
                    ),
                ],
            )
        ]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError()
