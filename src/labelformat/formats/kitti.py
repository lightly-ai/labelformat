import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

from labelformat import utils
from labelformat.cli.registry import Task, cli_register
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    SingleObjectDetection,
)

logger = logging.getLogger(__name__)


@cli_register(format="kitti", task=Task.OBJECT_DETECTION)
class KittiObjectDetectionInput(ObjectDetectionInput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-folder",
            type=Path,
            required=True,
            help="Input folder containing Kitti label txt files",
        )
        parser.add_argument(
            "--category-names",
            type=str,
            required=True,
            help="Comma separated list of category names without spaces, e.g. 'dog,cat'",
        )
        parser.add_argument(
            "--images-rel-path",
            type=str,
            default="../images",
            help="Relative path to images folder from label folder",
        )

    def __init__(
        self,
        input_folder: Path,
        category_names: str,
        images_rel_path: str = "../images",
    ) -> None:
        self._input_folder = input_folder
        self._images_rel_path = images_rel_path
        self._categories = [
            Category(id=idx, name=name)
            for idx, name in enumerate(category_names.split(","))
        ]

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def get_images(self) -> Iterable[Image]:
        yield from utils.get_images_from_folder(
            folder=self._input_folder / self._images_rel_path
        )

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        category_name_to_category = {cat.name: cat for cat in self._categories}

        for image in self.get_images():
            label_path = (self._input_folder / image.filename).with_suffix(".txt")
            if not label_path.exists():
                logger.warning(
                    f"Label file '{label_path}' for image '{image.filename}' does not exist."
                )

            objects = []
            with label_path.open() as file:
                for line in file.readlines():
                    # Last 14 tokens are floats. The rest in the beginning is a label.
                    tokens = line.split(" ")
                    category_name = " ".join(tokens[:-14])
                    left = float(tokens[-11])
                    top = float(tokens[-10])
                    right = float(tokens[-9])
                    bottom = float(tokens[-8])
                    objects.append(
                        SingleObjectDetection(
                            category=category_name_to_category[category_name],
                            box=BoundingBox.from_format(
                                bbox=[left, top, right, bottom],
                                format=BoundingBoxFormat.XYXY,
                            ),
                        )
                    )

            yield ImageObjectDetection(
                image=image,
                objects=objects,
            )


@cli_register(format="kitti", task=Task.OBJECT_DETECTION)
class KittiObjectDetectionOutput(ObjectDetectionOutput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help="Output folder for generated Kitti label txt files",
        )

    def __init__(
        self,
        output_folder: Path,
    ) -> None:
        self._output_folder = output_folder

    def save(self, label_input: ObjectDetectionInput) -> None:
        for image_label in label_input.get_labels():
            label_path = (self._output_folder / image_label.image.filename).with_suffix(
                ".txt"
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("w") as file:
                for obj in image_label.objects:
                    left, top, right, bottom = obj.box.to_format(
                        format=BoundingBoxFormat.XYXY
                    )
                    # Unknown values match Kitti dataset "DontCare" label values.
                    file.write(
                        f"{obj.category.name} "
                        "-1 "  # truncated
                        "-1 "  # occluded
                        "-10 "  # alpha
                        f"{left} {top} {right} {bottom} "  # bbox
                        "-1 -1 -1 "  # dimensions
                        "-1000 -1000 -1000 "  # location
                        "-10\n"  # rotation_y
                    )
