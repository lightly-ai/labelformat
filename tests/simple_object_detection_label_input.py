from labelformat.formats.labelformat import LabelformatObjectDetectionInput
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)


def get_input(
    filename: str = "image.jpg", with_confidence: bool = False
) -> LabelformatObjectDetectionInput:

    categories = [
        Category(id=0, name="cat"),
        Category(id=1, name="dog"),
        Category(id=2, name="cow"),
    ]
    images = [
        Image(id=0, filename=filename, width=100, height=200),
    ]
    labels = [
        ImageObjectDetection(
            image=images[0],
            objects=[
                SingleObjectDetection(
                    category=categories[1],
                    box=BoundingBox(
                        xmin=10.0,
                        ymin=20.0,
                        xmax=30.0,
                        ymax=40.0,
                    ),
                    confidence=0.4 if with_confidence else None,
                ),
                SingleObjectDetection(
                    category=categories[0],
                    box=BoundingBox(
                        xmin=50.0,
                        ymin=60.0,
                        xmax=70.0,
                        ymax=80.0,
                    ),
                    confidence=0.8 if with_confidence else None,
                ),
            ],
        )
    ]

    return LabelformatObjectDetectionInput(
        categories=categories,
        images=images,
        labels=labels,
    )
