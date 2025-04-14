from pathlib import Path

import pytest

from labelformat.formats.pascalvoc import (
    PascalVOCObjectDetectionInput,
    PascalVOCObjectDetectionOutput,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    SingleObjectDetection,
)

from ... import simple_object_detection_label_input


class TestPascalVOCObjectDetectionInput:
    def test_get_labels(self, tmp_path: Path) -> None:
        # Prepare inputs.
        annotation = """
            <annotation>
                <filename>image.jpg</filename>
                <size>
                    <width>100</width>
                    <height>200</height>
                </size>
                <object>
                    <name>dog</name>
                    <bndbox>
                        <xmin>10.0</xmin>
                        <ymin>20.0</ymin>
                        <xmax>30.0</xmax>
                        <ymax>40.0</ymax>
                    </bndbox>
                </object>
            </annotation>
        """
        xml_path = tmp_path / "labels" / "image.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(annotation)

        # Convert.
        label_input = PascalVOCObjectDetectionInput(
            input_folder=tmp_path / "labels", category_names="cat,dog"
        )
        labels = list(label_input.get_labels())
        assert labels == [
            ImageObjectDetection(
                image=Image(id=0, filename="image.jpg", width=100, height=200),
                objects=[
                    SingleObjectDetection(
                        category=Category(id=1, name="dog"),
                        box=BoundingBox(
                            xmin=10.0,
                            ymin=20.0,
                            xmax=30.0,
                            ymax=40.0,
                        ),
                    )
                ],
            )
        ]


class TestPascalVOCObjectDetectionOutput:
    @pytest.mark.parametrize("with_confidence", [True, False])
    def test_save(self, tmp_path: Path, with_confidence: bool) -> None:
        output_folder = tmp_path / "labels"
        PascalVOCObjectDetectionOutput(output_folder=output_folder).save(
            label_input=simple_object_detection_label_input.get_input(
                with_confidence=with_confidence
            )
        )
        assert output_folder.exists()
        assert output_folder.is_dir()

        filepaths = list(output_folder.glob("**/*"))
        assert len(filepaths) == 1
        path = filepaths[0]
        assert path == tmp_path / "labels" / "image.xml"

        contents = path.read_text()
        expected = """
            <annotation>
                <folder>labels</folder>
                <filename>image.jpg</filename>
                <size>
                    <width>100</width>
                    <height>200</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
                <object>
                    <name>dog</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <occluded>0</occluded>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>10.0</xmin>
                        <ymin>20.0</ymin>
                        <xmax>30.0</xmax>
                        <ymax>40.0</ymax>
                    </bndbox>
                </object>
                <object>
                    <name>cat</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <occluded>0</occluded>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>50.0</xmin>
                        <ymin>60.0</ymin>
                        <xmax>70.0</xmax>
                        <ymax>80.0</ymax>
                    </bndbox>
                </object>
            </annotation>
        """
        assert contents == expected.replace(" ", "").replace("\n", "")
