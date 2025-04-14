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


    Creation example:
    ```python
    from labelformat.formats.custom import CustomObjectDetectionInput
    from labelformat.model.bounding_box import BoundingBox
    from labelformat.model.category import Category
    from labelformat.model.image import Image
    from labelformat.model.object_detection import (
        ImageObjectDetection,
        SingleObjectDetection,
    )
    
    

    my_dataloader = ...
    my_model = ...
    prediction_bbox_format = "xywh" # or "xyxy" or "cxcywh", depending on your model
    
    categories = [
        Category(id=i, name=category)
        for i, category in enumerate(my_model.get_categories())
    ]

    images = []
    labels = []
    # Iterate over the PILImages and their filenames
    for i, image, filename in enumerate(my_dataloader):

        # Prediction for each image
        predictions = my_model.predict(image)
        
        # Create a new Image object for each image
        images.append(
            Image(id=i, filename=filename, width=image.width, height=image.height)
        )
        # Create a new ImageObjectDetection object for each image
        objects = []
        for prediction in predictions:
            objects.append(
                SingleObjectDetection(
                    category=categories[prediction.category],
                    box=BoundingBox.from_format(
                        box=prediction.box,
                        format=prediction_bbox_format,
                    ),
                    confidence=prediction.confidence,
                )
            )
        labels.append(
            ImageObjectDetection(
                image=images[-1],
                objects=objects,
            )
        )
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
