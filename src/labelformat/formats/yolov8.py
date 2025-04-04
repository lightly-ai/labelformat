import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List

import yaml

from labelformat import utils
from labelformat.cli.registry import Task, cli_register
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    InstanceSegmentationOutput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon, Point
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    SingleObjectDetection,
)
from labelformat.utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class _YOLOv8BaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file", type=Path, required=True, help="Path to data.yaml file"
        )
        parser.add_argument(
            "--input-split", type=str, default="train", help="Split to use"
        )

    def __init__(self, input_file: Path, input_split: str) -> None:
        self._config_file = input_file
        self._split = input_split
        with self._config_file.open() as file:
            self._config_data = yaml.safe_load(file)

        if self._split not in self._config_data:
            raise ValueError(
                f"Split '{self._split}' not found in config file '{self._config_file}'"
            )

    def get_categories(self) -> Iterable[Category]:
        """Get categories from YOLOv8 config file. Assumes contiguous 0-indexed labels."""
        names = self._config_data["names"]
        for category_id in range(len(names)):
            yield Category(id=category_id, name=names[category_id])

    def get_images(self) -> Iterable[Image]:
        yield from utils.get_images_from_folder(folder=self._images_dir())

    def _root_dir(self) -> Path:
        """Return the root directory of the dataset.

        If the config file contains a "path" field, it is used as the root directory (ultralytics format).
        Otherwise, the root directory is the parent of the config file (roboflow format).
        """
        if "path" in self._config_data:
            return self._config_file.parent / str(self._config_data["path"])
        return self._config_file.parent

    def _images_dir(self) -> Path:
        """Get images directory from YOLOv8 config file with fallback logic."""
        root_dir = self._root_dir()
        split_path = str(self._config_data[self._split])
        # Try original path first, then fallback to modified path for Roboflow-style configs
        path = root_dir / split_path
        if (
            not path.exists()
            and "path" not in self._config_data
            and split_path.startswith("../")
        ):
            split_path = split_path.replace("../", "./", 1)
            path = root_dir / split_path
        return path

    def _labels_dir(self) -> Path:
        """Get labels directory from YOLOv8 config file.

        The labels directory is derived from the images directory by replacing
        the first occurrence of 'images' with 'labels'.
        """
        root_dir = self._root_dir()
        images_dir = self._images_dir()
        images_dir_name = str(images_dir.relative_to(root_dir))

        if "images" not in images_dir_name:
            raise RuntimeError(
                f"Could not find 'images' subdirectory in '{images_dir}'"
            )

        labels_dir_name = images_dir_name.replace("images", "labels", 1)
        return root_dir / labels_dir_name


@cli_register(format="yolov8", task=Task.OBJECT_DETECTION)
class YOLOv8ObjectDetectionInput(_YOLOv8BaseInput, ObjectDetectionInput):
    """YOLOv8 format object detection dataset reader.

    Reads object detection annotations from a YOLOv8 dataset structure:
    - data.yaml: Contains dataset configuration and category mapping
    - images/: Contains input images
    - labels/: Contains annotation .txt files with normalized coordinates
    """

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        """Read object detection annotations from YOLOv8 format.

        Each .txt annotation file contains one object per line in format:
        <category_id> <center_x> <center_y> <width> <height>
        where coordinates are normalized to [0,1] range.

        Returns:
            Iterator of ImageObjectDetection containing the image metadata
            and its object annotations with absolute pixel coordinates.
        """
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        labels_dir = self._labels_dir()
        for image in self.get_images():
            label_path = (labels_dir / image.filename).with_suffix(".txt")
            if not label_path.exists():
                logger.warning(
                    f"Label file '{label_path}' for image '{image.filename}' does not exist. Skipping this image."
                )
                continue

            try:
                with label_path.open() as file:
                    label_data = [line.strip().split() for line in file if line.strip()]
            except OSError as e:
                logger.error(
                    f"Failed to access label file '{label_path}' for image '{image.filename}': {e}"
                )
                continue
            except ValueError as e:
                logger.error(
                    f"Error reading contents of label file '{label_path}' for image '{image.filename}': {e}"
                )
                continue

            objects = []
            for entry in label_data:
                if len(entry) != 5:
                    logger.warning(
                        f"Invalid label format in file '{label_path}' for image '{image.filename}'. Skipping this annotation."
                    )
                    continue

                try:
                    category_id, rcx, rcy, rw, rh = entry
                    cx = float(rcx) * image.width
                    cy = float(rcy) * image.height
                    w = float(rw) * image.width
                    h = float(rh) * image.height
                    objects.append(
                        SingleObjectDetection(
                            category=category_id_to_category[int(category_id)],
                            box=BoundingBox.from_format(
                                bbox=[cx, cy, w, h],
                                format=BoundingBoxFormat.CXCYWH,
                            ),
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.error(
                        f"Error processing annotation in file '{label_path}' for image '{image.filename}': {e}"
                    )
                    continue

            yield ImageObjectDetection(
                image=image,
                objects=objects,
            )


@cli_register(format="yolov8", task=Task.INSTANCE_SEGMENTATION)
class YOLOv8InstanceSegmentationInput(_YOLOv8BaseInput, InstanceSegmentationInput):
    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        labels_dir = self._labels_dir()
        for image in self.get_images():
            label_path = (labels_dir / image.filename).with_suffix(".txt")
            if not label_path.exists():
                logger.warning(
                    f"Label file '{label_path}' for image '{image.filename}' does not exist."
                )
            with label_path.open() as file:
                label_data = [line.split() for line in file.readlines()]

            objects = []
            for row in label_data:
                category_id, xy_points = row[0], row[1:]
                xs = [float(x) * image.width for x in xy_points[::2]]
                ys = [float(y) * image.height for y in xy_points[1::2]]
                objects.append(
                    SingleInstanceSegmentation(
                        category=category_id_to_category[int(category_id)],
                        segmentation=MultiPolygon(
                            polygons=[
                                list(zip(xs, ys)),
                            ],
                        ),
                    )
                )
            yield ImageInstanceSegmentation(
                image=image,
                objects=objects,
            )


class _YOLOv8BaseOutput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-file",
            type=Path,
            required=True,
            help="Output data.yaml file",
        )
        parser.add_argument(
            "--output-split",
            type=str,
            default="train",
            help="Split to use",
        )

    def __init__(
        self,
        output_file: Path,
        output_split: str,
    ) -> None:
        self._output_file = output_file
        self._output_split = output_split


@cli_register(format="yolov8", task=Task.OBJECT_DETECTION)
class YOLOv8ObjectDetectionOutput(_YOLOv8BaseOutput, ObjectDetectionOutput):
    def save(self, label_input: ObjectDetectionInput) -> None:
        # Write config file.
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        _save_dataset_yaml(
            output_file=self._output_file,
            output_split=self._output_split,
            categories=list(label_input.get_categories()),
        )

        # Write label files.
        labels_dir = self._output_file.parent / "labels"
        for label in label_input.get_labels():
            label_path = (labels_dir / label.image.filename).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("w") as file:
                for obj in label.objects:
                    cx, cy, w, h = obj.box.to_format(format=BoundingBoxFormat.CXCYWH)
                    rcx = cx / label.image.width
                    rcy = cy / label.image.height
                    rw = w / label.image.width
                    rh = h / label.image.height
                    file.write(f"{obj.category.id} {rcx} {rcy} {rw} {rh}\n")


@cli_register(format="yolov8", task=Task.INSTANCE_SEGMENTATION)
class YOLOv8InstanceSegmentationOutput(_YOLOv8BaseOutput, InstanceSegmentationOutput):
    def save(self, label_input: InstanceSegmentationInput) -> None:
        # Write config file.
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        _save_dataset_yaml(
            output_file=self._output_file,
            output_split=self._output_split,
            categories=list(label_input.get_categories()),
        )

        # Write label files.
        labels_dir = self._output_file.parent / "labels"
        for label in label_input.get_labels():
            label_path = (labels_dir / label.image.filename).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("w") as file:
                for obj in label.objects:
                    if not isinstance(obj.segmentation, MultiPolygon):
                        raise ValueError(
                            f"YOLOv8 format only supports MultiPolygon segmentation."
                        )
                    polygon = _multipolygon_to_polygon(multipolygon=obj.segmentation)
                    polygon_str = " ".join(
                        [
                            f"{x / label.image.width} {y / label.image.height}"
                            for x, y in polygon
                        ]
                    )
                    file.write(f"{obj.category.id} {polygon_str}\n")


def _save_dataset_yaml(
    output_file: Path,
    output_split: str,
    categories: List[Category],
) -> None:
    with output_file.open("w") as file:
        yaml.dump(
            {
                "path": ".",
                output_split: "images",
                "nc": len(categories),
                "names": {category.id: category.name for category in categories},
            },
            file,
        )


def _multipolygon_to_polygon(
    multipolygon: MultiPolygon,
) -> List[Point]:
    """Convert MultiPolygon to polygon.

    YOLOv8 segmentation format only supports single polygons. To convert
    MultiPolygon to a single polygon, we take the first point of every polygon
    and create a zero-width link with the first point of the next polygon.

    Note: This approach might create self-intersecting polygons and may cause issues
    e.g. if the representation mixes clockwise and counter-clockwise polygons.
    """
    if len(multipolygon.polygons) == 0:
        raise ValueError("Cannot convert empty MultiPolygon to polygon.")

    # First add all the polygons. Note we add the first point of every polygon
    # also at the end to close it.
    out_polygon = []
    for polygon in multipolygon.polygons:
        out_polygon.extend(polygon)
        out_polygon.append(polygon[0])

    # The 'cursor' is now at the end (first) point of the last polygon. We need to close
    # the polygon by adding the first points of all the other polygons in reverse order.
    for polygon in reversed(multipolygon.polygons[:-1]):
        out_polygon.append(polygon[0])

    return out_polygon
