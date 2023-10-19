import copy
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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
from labelformat.types import JsonDict


@cli_register(format="lightly", task=Task.OBJECT_DETECTION)
class LightlyObjectDetectionInput(ObjectDetectionInput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-folder",
            type=Path,
            required=True,
            help="Path to input folder with JSON files",
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
        images_rel_path: str = "../images",
    ) -> None:
        self._input_folder = input_folder
        self._images_rel_path = images_rel_path
        self._categories = self._get_categories()

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def get_images(self) -> Iterable[Image]:
        yield from utils.get_images_from_folder(
            folder=self._input_folder / self._images_rel_path
        )

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        filename_to_image = {image.filename: image for image in self.get_images()}

        for json_path in self._input_folder.glob("*.json"):
            if json_path.name == "schema.json":
                continue
            data = json.loads(json_path.read_text())
            image = filename_to_image[data["file_name"]]
            objects = []
            for prediction in data["predictions"]:
                objects.append(
                    SingleObjectDetection(
                        category=category_id_to_category[prediction["category_id"]],
                        box=BoundingBox.from_format(
                            bbox=[float(x) for x in prediction["bbox"]],
                            format=BoundingBoxFormat.XYWH,
                        ),
                    )
                )
            yield ImageObjectDetection(
                image=image,
                objects=objects,
            )

    def _get_categories(self) -> Sequence[Category]:
        schema_path = self._input_folder / "schema.json"
        schema_json = json.loads(schema_path.read_text())
        if schema_json["task_type"] != "object-detection":
            raise ValueError(
                f"Schema type '{schema_json['task_type']}' is not supported. "
                f"Expected 'object-detection'."
            )
        return [
            Category(
                id=category["id"],
                name=category["name"],
            )
            for category in schema_json["categories"]
        ]


@cli_register(format="lightly", task=Task.OBJECT_DETECTION)
class LightlyObjectDetectionOutput(ObjectDetectionOutput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help=(
                "Path to output folder with JSON files. The folder name should "
                "match the detection task name."
            ),
        )

    def __init__(self, output_folder: Path) -> None:
        self._output_folder = output_folder

    def save(self, label_input: ObjectDetectionInput) -> None:
        self._output_folder.mkdir(parents=True, exist_ok=True)

        # Save schema.
        schema = {
            "task_type": "object-detection",
            "categories": [
                {
                    "id": category.id,
                    "name": category.name,
                }
                for category in label_input.get_categories()
            ],
        }
        schema_file = self._output_folder / "schema.json"
        with schema_file.open("w") as file:
            json.dump(schema, file, indent=2)

        # Save labels.
        for label in label_input.get_labels():
            data = {
                "file_name": label.image.filename,
                "predictions": [
                    {
                        "category_id": obj.category.id,
                        "bbox": obj.box.to_format(BoundingBoxFormat.XYWH),
                        "score": 0.0,  # default
                    }
                    for obj in label.objects
                ],
            }
            label_file = (self._output_folder / f"{label.image.filename}").with_suffix(
                ".json"
            )
            with label_file.open("w") as file:
                json.dump(data, file, indent=2)
