import logging
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
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
        self._categories = self._get_categories()

    def get_categories(self) -> Iterable[Category]:
        return self._categories

    def _get_categories(self) -> List[Category]:
        """Get categories from XML by searching in possible label paths."""
        label_paths = ["meta/task/labels", "meta/job/labels", "meta/project/labels"]
        
        xml_labels = next(
            (self._data.find(path) for path in label_paths if self._data.find(path) is not None),
            None,
        )
        
        if xml_labels is None:
            raise ParseError(
                f"Could not find labels at any of the provided paths: {', '.join(label_paths)}"
            )
        
        return [
            Category(
                id=idx,
                name=CVATParser._xml_text_or_raise(CVATParser._xml_find_or_raise(label, "name"))
            )
            for idx, label in enumerate(xml_labels.findall("label"))
        ]


class CVATParser:
    """Handles parsing of CVAT XML elements into domain objects."""
    
    def __init__(self, categories: List[Category]):
        self.categories = categories

    def parse_image(self, xml_root: ET.Element) -> Image:
        self._validate_required_attributes(xml_root, ["name", "id", "width", "height"])
        
        return Image(
            id=int(self._xml_attribute_text_or_raise(xml_root, "id")),
            filename=self._xml_attribute_text_or_raise(xml_root, "name"),
            width=int(self._xml_attribute_text_or_raise(xml_root, "width")),
            height=int(self._xml_attribute_text_or_raise(xml_root, "height")),
        )

    def parse_objects(self, xml_root: ET.Element) -> List[SingleObjectDetection]:
        """Parse bounding box objects from XML and return list of detections."""
        required_box_attributes = ["label", "xtl", "ytl", "xbr", "ybr"]
        
        return [
            SingleObjectDetection(
                category=self._get_category_from_label(xml_box.get("label")),
                box=BoundingBox.from_format(
                    bbox=self._extract_bbox_coordinates(
                        self._validate_required_attributes(xml_box, required_box_attributes) or xml_box
                    ),
                    format=BoundingBoxFormat.XYXY,
                ),
            )
            for xml_box in xml_root.findall("box")
        ]

    def _get_category_from_label(self, label: str) -> Category:
        """Find matching category for label name or raise error."""
        category = next((cat for cat in self.categories if cat.name == label), None)
        if category is None:
            raise ParseError(f"Unknown category name '{label}'.")
        return category

    def _extract_bbox_coordinates(self, xml_box: ET.Element) -> List[float]:
        """Extract and convert bounding box coordinates from XML element."""
        coordinate_attrs = ["xtl", "ytl", "xbr", "ybr"]
        return [
            float(self._xml_attribute_text_or_raise(xml_box, attr))
            for attr in coordinate_attrs
        ]

    @staticmethod
    def _xml_attribute_text_or_raise(elem: ET.Element, attribute_name: str) -> str:
        attribute_text = elem.get(attribute_name)
        if attribute_text is None:
            raise ParseError(f"Could not read attribute: '{attribute_name}'")
        return attribute_text

    @staticmethod
    def _validate_required_attributes(
        xml_elem: ET.Element, required_attributes: List[str]
    ) -> None:
        missing_attrs = [attr for attr in required_attributes if xml_elem.get(attr) is None]
        if missing_attrs:
            raise ParseError(f"Missing required attributes: {', '.join(missing_attrs)}")

    @staticmethod
    def _xml_find_or_raise(elem: ET.Element, path: str) -> ET.Element:
        found_elem = elem.find(path=path)
        if found_elem is None:
            raise ParseError(f"Missing field '{path}' in XML.")
        return found_elem

    @staticmethod
    def _xml_text_or_raise(elem: ET.Element) -> str:
        text = elem.text
        if text is None:
            raise ParseError(
                f"Missing text content for XML element: {ET.tostring(elem, encoding='unicode')}"
            )
        return text


@cli_register(format="cvat", task=Task.OBJECT_DETECTION)
class CVATObjectDetectionInput(_CVATBaseInput, ObjectDetectionInput):
    def __init__(self, input_file: Path) -> None:
        super().__init__(input_file)
        self._parser = CVATParser(self._categories)

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        """Yield ImageObjectDetection instances from XML data."""
        for xml_image in self._data.findall("image"):
            try:
                yield ImageObjectDetection(
                    image=self._parser.parse_image(xml_root=xml_image),
                    objects=self._parser.parse_objects(xml_root=xml_image),
                )
            except ParseError as ex:
                raise ParseError(f"Could not parse XML file: {str(ex)}") from ex


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
        self._add_meta_labels(root, label_input.get_categories())
        
        # Add image annotations
        self._add_image_annotations(root, label_input.get_labels())

        # Save XML file
        tree = ET.ElementTree(root)
        label_path = (self._output_folder / "annotations").with_suffix(".xml")
        tree.write(
            label_path,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False,
        )

    def _add_meta_labels(self, root: ET.Element, categories: Iterable[Category]) -> None:
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, self._annotation_scope)
        labels = ET.SubElement(task, "labels")

        for category in categories:
            label = ET.SubElement(labels, "label")
            name = ET.SubElement(label, "name")
            name.text = category.name

    def _add_image_annotations(
        self, root: ET.Element, labels: Iterable[ImageObjectDetection]
    ) -> None:
        for label_object in labels:
            image_elem = self._create_image_element(root, label_object.image)
            self._add_bounding_boxes(image_elem, label_object.objects)

    def _create_image_element(self, root: ET.Element, image: Image) -> ET.Element:
        return ET.SubElement(
            root,
            "image",
            {
                "id": str(image.id),
                "name": image.filename,
                "width": str(image.width),
                "height": str(image.height),
            },
        )

    def _add_bounding_boxes(
        self, image_elem: ET.Element, objects: List[SingleObjectDetection]
    ) -> None:
        """Add bounding box elements to the image element."""
        [
            ET.SubElement(
                image_elem,
                "box",
                {
                    "label": obj.category.name,
                    "xtl": str(obj.box.xmin),
                    "ytl": str(obj.box.ymin),
                    "xbr": str(obj.box.xmax),
                    "ybr": str(obj.box.ymax),
                },
            )
            for obj in objects
        ]


