import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pytest

from labelformat.formats.cvat import (
    AnnotationScope,
    CVATObjectDetectionInput,
    CVATObjectDetectionOutput,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)
from labelformat.types import ParseError


# Helper for creating temp XML files
def create_xml_file(tmp_path: Path, content: str) -> Path:
    xml_path = tmp_path / "labels" / "annotations_in.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(content.strip())
    return xml_path


class TestCVATObjectDetectionInput:
    @pytest.mark.parametrize(
        "annotation_scope",
        [AnnotationScope.TASK, AnnotationScope.PROJECT, AnnotationScope.JOB],
    )
    def test_get_labels(
        self, tmp_path: Path, annotation_scope: AnnotationScope
    ) -> None:
        annotation = f"""
          <annotations>
            <version>1.1</version>
            <meta>
                <{annotation_scope.value}>
                  <labels>
                    <label><name>label1</name></label>
                    <label><name>label2</name></label>
                  </labels>
                </{annotation_scope.value}>
            </meta>
            <image height="8" id="0" name="img0.jpg" width="10">
                <box label="label1" occluded="0" xtl="4" ytl="0" xbr="4" ybr="2" z_order="1"></box>
            </image>
            <version>1.1</version>
          </annotations>
        """

        xml_path = create_xml_file(tmp_path, annotation)
        label_input = CVATObjectDetectionInput(xml_path)

        # Validate categories.
        categories = list(label_input.get_categories())
        assert categories == [
            Category(id=1, name="label1"),
            Category(id=2, name="label2"),
        ]

        # Validate labels.
        labels = list(label_input.get_labels())
        assert labels == [
            ImageObjectDetection(
                image=Image(id=0, filename="img0.jpg", width=10, height=8),
                objects=[
                    SingleObjectDetection(
                        category=Category(id=1, name="label1"),
                        box=BoundingBox(xmin=4.0, ymin=0.0, xmax=4.0, ymax=2.0),
                    )
                ],
            )
        ]

    def test___init___invalid_xml(self, tmp_path: Path) -> None:
        invalid_annotation = """
          <annotations>
              <meta>
                <task>
                  <labels>
                    <label><name>label1</name></label>
                  </labels>
                </task>
              </meta>
              <image id="0" name="img0.jpg" width="10" height="8">
                <box label="label1" xtl="invalid" ytl="0.0" xbr="5.0" ybr="2.0"></box>
              </image>
          </annotations>
        """
        xml_path = create_xml_file(tmp_path, invalid_annotation)

        with pytest.raises(
            ValueError,
            match="Input should be a valid number, unable to parse string as a number",
        ):
            label_input = CVATObjectDetectionInput(xml_path)

    def test___init____missing_attributes_for_image(self, tmp_path: Path) -> None:
        invalid_annotation = """
          <annotations>
              <meta>
                <task>
                  <labels>
                    <label><name>label1</name></label>
                  </labels>
                </task>
              </meta>
              <image id="0" name="img0.jpg" width="10" >
                <box label="label1" xtl="0.0" ytl="0.0" xbr="5.0" ybr="2.0"></box>
              </image>
          </annotations>
        """
        xml_path = create_xml_file(tmp_path, invalid_annotation)

        with pytest.raises(
            ValueError,
            match="validation error for CVATAnnotations\nimages.0.height",
        ):
            label_input = CVATObjectDetectionInput(xml_path)

    def test_get_labels_invalid_category_name(self, tmp_path: Path) -> None:
        invalid_annotation = """
          <annotations>
              <meta>
                <task>
                  <labels>
                    <label><name>label1</name></label>
                  </labels>
                </task>
              </meta>
              <image id="0" name="img0.jpg" width="10" height="8">
                <box label="label2" xtl="1.0" ytl="0.0" xbr="5.0" ybr="2.0"></box>
              </image>
          </annotations>
        """
        xml_path = create_xml_file(tmp_path, invalid_annotation)

        with pytest.raises(ParseError, match="Unknown category name 'label2'"):
            label_input = CVATObjectDetectionInput(xml_path)
            list(label_input.get_labels())


def _compare_xml_elements(elem1: ET.Element, elem2: ET.Element) -> bool:
    """Recursively compare two XML elements for tag, attributes, and text."""

    def normalize(text: Optional[str]) -> str:
        return (text or "").strip().replace("\n", "").replace(" ", "")

    if elem1.tag != elem2.tag or normalize(elem1.text) != normalize(elem2.text):
        return False

    if elem1.attrib != elem2.attrib:
        return False

    children1 = list(elem1)
    children2 = list(elem2)

    if len(children1) != len(children2):
        return False

    return all(_compare_xml_elements(c1, c2) for c1, c2 in zip(children1, children2))


class TestCVATObjectDetectionOutput:
    @pytest.mark.parametrize(
        "annotation_scope",
        [AnnotationScope.TASK, AnnotationScope.PROJECT, AnnotationScope.JOB],
    )
    def test_save_cyclic_load_and_save(
        self, tmp_path: Path, annotation_scope: AnnotationScope
    ) -> None:
        annotation = f"""<annotations>
              <meta>
                <{annotation_scope.value}>
                  <labels>
                    <label><name>label1</name></label>
                    <label><name>label2</name></label>
                  </labels>
                </{annotation_scope.value}>
              </meta>
              <image id="0" name="img0.jpg" width="10" height="8">
                <box label="label1" xtl="4.0" ytl="0.0" xbr="5.0" ybr="2.0"></box>
              </image>
          </annotations>
        """

        input_xml_path = create_xml_file(tmp_path, annotation)
        label_input = CVATObjectDetectionInput(input_xml_path)
        output_folder = tmp_path / "labels"

        CVATObjectDetectionOutput(
            output_folder=output_folder, output_annotation_scope=annotation_scope.value
        ).save(label_input=label_input)

        assert output_folder.exists()
        assert output_folder.is_dir()
        filepaths = list(output_folder.glob("**/*.xml"))
        assert len(filepaths) == 2

        output_xml_path = tmp_path / "labels" / "annotations.xml"
        # Compare XML structure.
        input_tree = ET.parse(input_xml_path)
        output_tree = ET.parse(output_xml_path)

        assert _compare_xml_elements(
            input_tree.getroot(), output_tree.getroot()
        ), "The output XML structure doesn't match the input XML."
