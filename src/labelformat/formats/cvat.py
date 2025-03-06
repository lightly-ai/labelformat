import logging
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

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
from labelformat.types import ParseError

logger = logging.getLogger(__name__)


class _CVATBaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to input CVAT XML annotations file",
        )

    def __init__(self, input_file: Path) -> None:
        try:
            self._data = ET.parse(input_file).getroot()
        except ET.ParseError as ex:
            raise ParseError(
                f"Could not parse XML file {input_file}: {str(ex)}"
            ) from ex
        self._categories = _get_categories(self._data)

    def get_categories(self) -> Iterable[Category]:
        return self._categories


@cli_register(format="cvat", task=Task.OBJECT_DETECTION)
class CVATObjectDetectionInput(_CVATBaseInput, ObjectDetectionInput):
    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        xml_images = self._data.findall("image")
        for xml_image in xml_images:
            try:
                image = _parse_image(
                    xml_root=xml_image,
                )
                objects = _parse_object(
                    xml_root=xml_image,
                    categories=self._categories,
                )
            except ParseError as ex:
                raise ParseError(f"Could not parse XML file : {str(ex)}") from ex

            yield ImageObjectDetection(
                image=image,
                objects=objects,
            )


@cli_register(format="cvat", task=Task.OBJECT_DETECTION)
class CVATObjectDetectionOutput(ObjectDetectionOutput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help="Output folder to store generated CVAT XML annotations file",
        )
        parser.add_argument(
            "--output-annotation-scope ",
            choices=["task", "job", "project"],
            default="task",
            help="Define the annotation scope to determine the XML structure: 'task', 'job', or 'project'.",
        )

    def __init__(self, output_folder: Path, annotation_scope: str) -> None:
        self._output_folder = output_folder
        self._annotation_scope = annotation_scope

    def save(self, label_input: ObjectDetectionInput) -> None:
        # Write config file.
        self._output_folder.mkdir(parents=True, exist_ok=True)
        root = ET.Element("annotations")

        # Add meta information with labels
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, self._annotation_scope)
        labels = ET.SubElement(task, "labels")

        # Adding categories as labels
        for category in label_input.get_categories():
            label = ET.SubElement(labels, "label")
            name = ET.SubElement(label, "name")
            name.text = category.name

        for label_object in label_input.get_labels():
            image_elem = ET.SubElement(
                root,
                "image",
                OrderedDict(
                    [
                        ("id", str(label_object.image.id)),
                        ("name", label_object.image.filename),
                        ("width", str(label_object.image.width)),
                        ("height", str(label_object.image.height)),
                    ]
                ),
            )

            for obj in label_object.objects:
                bbox = obj.box
                ET.SubElement(
                    image_elem,
                    "box",
                    {
                        "label": obj.category.name,
                        "xtl": str(bbox.xmin),
                        "ytl": str(bbox.ymin),
                        "xbr": str(bbox.xmax),
                        "ybr": str(bbox.ymax),
                    },
                )

        tree = ET.ElementTree(root)
        label_path = (self._output_folder / "annotations").with_suffix(".xml")
        tree.write(
            label_path,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False,
        )


def _get_categories(xml_root: ET.Element) -> List[Category]:
    label_paths = ["meta/task/labels", "meta/job/labels", "meta/project/labels"]
    for path in label_paths:
        xml_labels = xml_root.find(path)
        if xml_labels is not None:
            xml_objects = xml_labels.findall("label")
            categories = [
                Category(
                    id=index, name=_xml_text_or_raise(_xml_find_or_raise(label, "name"))
                )
                for index, label in enumerate(xml_objects)
            ]
            return categories
    raise ParseError(
        f"Could not find labels at any of the provided paths: {', '.join(label_paths)}"
    )


def _parse_image(xml_root: ET.Element) -> Image:
    _validate_required_attributes(xml_root, ["name", "id", "width", "height"])

    return Image(
        id=int(_xml_attribute_text_or_raise(xml_root, "id")),
        filename=_xml_attribute_text_or_raise(xml_root, "name"),
        width=int(_xml_attribute_text_or_raise(xml_root, "width")),
        height=int(_xml_attribute_text_or_raise(xml_root, "height")),
    )


def _parse_object(
    categories: List[Category], xml_root: ET.Element
) -> List[SingleObjectDetection]:
    objects = []
    xml_boxes = xml_root.findall("box")
    for xml_box in xml_boxes:
        _validate_required_attributes(xml_box, ["label", "xtl", "ytl", "xbr", "ybr"])

        label = xml_box.get("label")
        category = next((cat for cat in categories if cat.name == label), None)
        if category is None:
            raise ParseError(f"Unknown category name '{label}'.")
        bbox = [
            float(_xml_attribute_text_or_raise(xml_box, attr))
            for attr in ["xtl", "ytl", "xbr", "ybr"]
        ]

        objects.append(
            SingleObjectDetection(
                category=category,
                box=BoundingBox.from_format(
                    bbox=bbox,
                    format=BoundingBoxFormat.XYXY,
                ),
            )
        )

    return objects


def _xml_find_or_raise(elem: ET.Element, path: str) -> ET.Element:
    found_elem = elem.find(path=path)
    if found_elem is None:
        raise ParseError(f"Missing field '{path}' in XML.")
    return found_elem


def _xml_text_or_raise(elem: ET.Element) -> str:
    text = elem.text
    if text is None:
        raise ParseError(
            f"Missing text content for XML element: {ET.tostring(elem, encoding='unicode')}"
        )
    return text


def _xml_attribute_text_or_raise(elem: ET.Element, attribute_name: str) -> str:
    attribute_text = elem.get(attribute_name)
    if attribute_text is None:
        raise ParseError(f"Bad value for attribute: {attribute_name}")
    return attribute_text


def _validate_required_attributes(
    xml_elem: ET.Element, required_attributes: List[str]
) -> None:
    missing_attrs = [attr for attr in required_attributes if xml_elem.get(attr) is None]
    if missing_attrs:
        raise ParseError(f"Missing required attributes: {', '.join(missing_attrs)}")
