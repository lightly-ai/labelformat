import json
import logging
import re
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
from labelformat.types import JsonDict, ParseError

logger = logging.getLogger(__name__)


class FilenameKeyOption(Enum):
    GLOBAL_KEY = "global_key"
    EXTERNAL_ID = "external_id"
    ID = "id"

    def __str__(self) -> str:
        """Required for a user-friendly string representation for the CLI."""
        return self.value


@cli_register(format="labelbox", task=Task.OBJECT_DETECTION)
class LabelboxObjectDetectionInput(ObjectDetectionInput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to Labelbox export v2 ndjson file",
        )
        parser.add_argument(
            "--category-names",
            type=str,
            required=True,
            help="Comma separated list of category names without spaces, e.g. 'dog,cat'",
        )
        parser.add_argument(
            "--filename-key",
            type=FilenameKeyOption,
            choices=list(FilenameKeyOption),
            default=FilenameKeyOption.GLOBAL_KEY,
            help=(
                "Which Labelbox json key should be used as exported image filename. "
                "Default: global_key"
            ),
        )

    def __init__(
        self,
        input_file: Path,
        category_names: str,
        filename_key: FilenameKeyOption = FilenameKeyOption.GLOBAL_KEY,
    ) -> None:
        self._input_file = input_file
        self._categories = [
            Category(id=idx, name=name)
            for idx, name in enumerate(category_names.split(","))
        ]
        self._filename_key = filename_key

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        category_name_to_category = {cat.name: cat for cat in self._categories}

        with self._input_file.open() as file:
            image_id = 0
            for line in file:
                data_row = json.loads(line)

                try:
                    yield _parse_data_row(
                        category_name_to_category=category_name_to_category,
                        image_id=image_id,
                        data_row=data_row,
                        filename_key=self._filename_key,
                    )
                except ParseError as ex:
                    raise ParseError(
                        f"Could not parse data row {line}: {str(ex)}"
                    ) from ex

                image_id += 1


def _parse_data_row(
    category_name_to_category: Dict[str, Category],
    image_id: int,
    data_row: JsonDict,
    filename_key: FilenameKeyOption,
) -> ImageObjectDetection:
    image = _image_from_data_row(
        image_id=image_id, data_row=data_row, filename_key=filename_key
    )
    objects = _objects_from_data_row(
        category_name_to_category=category_name_to_category,
        data_row=data_row,
    )
    return ImageObjectDetection(image=image, objects=objects)


def _has_illegal_char(filename: str) -> bool:
    """Checks if filename contains illegal characters for filenames"""
    return bool(re.search(r'[\\/*?:"<>|]', filename))


def _image_from_data_row(
    image_id: int, data_row: JsonDict, filename_key: FilenameKeyOption
) -> Image:
    if filename_key.value not in data_row["data_row"]:
        raise ParseError(
            f"Filename key '{filename_key.value}' not found in data_row. Consider "
            f"choosing a different key from {[e.value for e in FilenameKeyOption]}. "
            f"Data row: {data_row['data_row']}"
        )

    filename = data_row["data_row"][filename_key.value]
    if _has_illegal_char(filename=filename):
        raise ParseError(
            f"Filename key '{filename_key.value}' cannot be used because one of the "
            f"values '{filename}' contains illegal characters. Please choose a "
            f"different key from {[e.value for e in FilenameKeyOption]}."
        )

    width = data_row["media_attributes"]["width"]
    height = data_row["media_attributes"]["height"]

    return Image(
        id=image_id,
        filename=filename,
        width=width,
        height=height,
    )


def _objects_from_data_row(
    category_name_to_category: Dict[str, Category],
    data_row: JsonDict,
) -> List[SingleObjectDetection]:
    if len(data_row["projects"]) != 1:
        raise ParseError(
            "Labelbox format reader currently supports only a single project. "
            f"Found {len(data_row['projects'])} projects in the export."
        )
    project = next(iter(data_row["projects"].values()))

    if len(project["labels"]) != 1:
        raise ParseError(
            "Labelbox format reader currently expects a single entry "
            "in the 'labels' list."
        )
    annotations = project["labels"][0]["annotations"]

    if "frames" in annotations:
        raise ParseError(
            "Found 'frames' in annotations. "
            "Labelbox format reader currently does not support videos."
        )

    objects = []
    for obj in annotations["objects"]:
        if obj["annotation_kind"] != "ImageBoundingBox":
            logger.warning(
                f"Skipping object with annotation_kind '{obj['annotation_kind']}'. "
                "Only ImageBoundingBox is supported."
            )
            continue

        category_name = obj["name"]
        if category_name not in category_name_to_category:
            raise ParseError(f"Unknown category name '{category_name}'.")
        category = category_name_to_category[category_name]

        bbox = obj["bounding_box"]
        objects.append(
            SingleObjectDetection(
                category=category,
                box=BoundingBox.from_format(
                    bbox=[
                        float(bbox["left"]),
                        float(bbox["top"]),
                        float(bbox["width"]),
                        float(bbox["height"]),
                    ],
                    format=BoundingBoxFormat.XYWH,
                ),
            )
        )

    return objects
