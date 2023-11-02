import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List

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

    def __init__(
        self,
        input_file: Path,
        category_names: str,
    ) -> None:
        self._input_file = input_file
        self._categories = [
            Category(id=idx, name=name)
            for idx, name in enumerate(category_names.split(","))
        ]

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self, filename_key: str = None) -> Iterable[ImageObjectDetection]:
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
                        filename_key=filename_key,
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
    filename_key: str = None,
) -> ImageObjectDetection:
    image = _image_from_data_row(image_id=image_id, data_row=data_row, filename_key=filename_key)
    objects = _objects_from_data_row(
        category_name_to_category=category_name_to_category,
        data_row=data_row,
    )
    return ImageObjectDetection(image=image, objects=objects)


def has_illegal_char(filename: str) -> bool:
    """Checks if filename contains illegal characters for filenames"""
    return bool(re.search(r'[\\/*?:"<>|]', filename))


def _image_from_data_row(image_id: int, data_row: JsonDict, filename_key: str = None) -> Image:
    fn_keys = ["global_key", "external_id", "id"]
    if filename_key is not None:
        # use filename_key if it is valid
        assert filename_key in fn_keys, \
            f"Filename key '{filename_key}' is not a valid option, please use one of {fn_keys}"
        assert has_illegal_char(data_row["data_row"][filename_key]) == False, \
            (f"Filename key '{filename_key}' contains illegal characters for filenames, "
             f"please use another option among {fn_keys}")
        assert filename_key in data_row["data_row"], \
            f"Filename key '{filename_key}' does not exist in data_row: {data_row['data_row']}"
        filename = data_row["data_row"][filename_key]
    else:
        # try to use one of fn_keys as filename
        for key in fn_keys:
            if key in data_row["data_row"] and not has_illegal_char(data_row["data_row"][key]):
                filename = data_row["data_row"][key]
                break
        else:
            raise ParseError(
                f"Could not parse image filename from data row: {data_row['data_row']}. None of 'data_row.id', "
                f"'data_row.global_key' nor 'data_row.external_id' existed or in the legal format for a filename."
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
