import logging
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List

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


@cli_register(format="pascalvoc", task=Task.OBJECT_DETECTION)
class PascalVOCObjectDetectionInput(ObjectDetectionInput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-folder",
            type=Path,
            required=True,
            help="Input folder containing PascalVOC XML files",
        )
        parser.add_argument(
            "--category-names",
            type=str,
            required=True,
            help="Comma separated list of category names without spaces, e.g. 'dog,cat'",
        )

    def __init__(self, input_folder: Path, category_names: str) -> None:
        if not input_folder.is_dir():
            raise ValueError(f"Input folder '{input_folder}' is not a directory.")
        self._input_folder = input_folder
        self._categories = [
            Category(id=idx, name=name)
            for idx, name in enumerate(category_names.split(","))
        ]

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        category_name_to_category = {cat.name: cat for cat in self._categories}

        image_id = 0
        for xml_path in self._input_folder.glob("*.xml"):
            xml_root = ET.parse(xml_path).getroot()

            try:
                image = _parse_image(
                    image_id=image_id,
                    xml_root=xml_root,
                )
                objects = _parse_objects(
                    xml_root=xml_root,
                    category_name_to_category=category_name_to_category,
                )
            except ParseError as ex:
                raise ParseError(
                    f"Could not parse XML file {xml_path}: {str(ex)}"
                ) from ex

            yield ImageObjectDetection(
                image=image,
                objects=objects,
            )
            image_id += 1


@cli_register(format="pascalvoc", task=Task.OBJECT_DETECTION)
class PascalVOCObjectDetectionOutput(ObjectDetectionOutput):
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help="Output folder to store generated PascalVOC XML files",
        )

    def __init__(self, output_folder: Path) -> None:
        self._output_folder = output_folder

    def save(self, label_input: ObjectDetectionInput) -> None:
        # Write config file.
        self._output_folder.mkdir(parents=True, exist_ok=True)

        for label in label_input.get_labels():
            annotation_elem = _create_annotation(
                label=label,
                folder_name=self._output_folder.name,
            )
            xml_tree = ET.ElementTree(annotation_elem)
            label_path = (self._output_folder / label.image.filename).with_suffix(
                ".xml"
            )
            xml_tree.write(label_path)


def _parse_image(image_id: int, xml_root: ET.Element) -> Image:
    filename_elem = _xml_find_or_raise(elem=xml_root, path="filename")
    size_elem = _xml_find_or_raise(elem=xml_root, path="size")
    width_elem = _xml_find_or_raise(elem=size_elem, path="width")
    height_elem = _xml_find_or_raise(elem=size_elem, path="height")

    filename = _xml_text_or_raise(elem=filename_elem)
    width = int(_xml_text_or_raise(elem=width_elem))
    height = int(_xml_text_or_raise(elem=height_elem))
    return Image(id=image_id, filename=filename, width=width, height=height)


def _parse_objects(
    category_name_to_category: Dict[str, Category], xml_root: ET.Element
) -> List[SingleObjectDetection]:
    xml_objects = xml_root.findall("object")
    objects = []
    for xml_obj in xml_objects:
        name_elem = _xml_find_or_raise(elem=xml_obj, path="name")
        name = _xml_text_or_raise(elem=name_elem)
        if name not in category_name_to_category:
            raise ParseError(f"Unknown category name '{name}'.")
        category = category_name_to_category[name]

        bndbox = _xml_find_or_raise(elem=xml_obj, path="bndbox")
        xmin_elem = _xml_find_or_raise(elem=bndbox, path="xmin")
        ymin_elem = _xml_find_or_raise(elem=bndbox, path="ymin")
        xmax_elem = _xml_find_or_raise(elem=bndbox, path="xmax")
        ymax_elem = _xml_find_or_raise(elem=bndbox, path="ymax")
        xmin = float(_xml_text_or_raise(elem=xmin_elem))
        ymin = float(_xml_text_or_raise(elem=ymin_elem))
        xmax = float(_xml_text_or_raise(elem=xmax_elem))
        ymax = float(_xml_text_or_raise(elem=ymax_elem))

        objects.append(
            SingleObjectDetection(
                category=category,
                box=BoundingBox.from_format(
                    bbox=[xmin, ymin, xmax, ymax],
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


def _create_annotation(
    label: ImageObjectDetection,
    folder_name: str,
) -> ET.Element:
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = folder_name

    filename = ET.SubElement(annotation, "filename")
    filename.text = label.image.filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(label.image.width)
    ET.SubElement(size, "height").text = str(label.image.height)
    ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

    ET.SubElement(annotation, "segmented").text = "0"  # default value

    for obj in label.objects:
        bbox_vals = obj.box.to_format(BoundingBoxFormat.XYXY)
        xml_obj = ET.SubElement(annotation, "object")
        ET.SubElement(xml_obj, "name").text = obj.category.name
        ET.SubElement(xml_obj, "pose").text = "Unspecified"
        ET.SubElement(xml_obj, "truncated").text = "0"  # default value
        ET.SubElement(xml_obj, "occluded").text = "0"  # default value
        ET.SubElement(xml_obj, "difficult").text = "0"  # default value

        bndbox = ET.SubElement(xml_obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox_vals[0])
        ET.SubElement(bndbox, "ymin").text = str(bbox_vals[1])
        ET.SubElement(bndbox, "xmax").text = str(bbox_vals[2])
        ET.SubElement(bndbox, "ymax").text = str(bbox_vals[3])

    return annotation
