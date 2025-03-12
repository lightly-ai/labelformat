import logging
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic_xml import BaseXmlModel, attr, element

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


# The following Pydantic XML models describe the structure of CVAT XML files.
class CVATLabel(BaseXmlModel, tag="label", search_mode="unordered"):  # type: ignore
    name: str = element()


class CVATLabels(BaseXmlModel, tag="labels", search_mode="unordered"):  # type: ignore
    label_list: List[CVATLabel] = element(tag="label")


class CVATTask(BaseXmlModel, tag="task", search_mode="unordered"):  # type: ignore
    labels: Optional[CVATLabels] = element(tag="labels")


class CVATJob(BaseXmlModel, tag="job", search_mode="unordered"):  # type: ignore
    labels: Optional[CVATLabels] = element(tag="labels")


class CVATProject(BaseXmlModel, tag="project", search_mode="unordered"):  # type: ignore
    labels: Optional[CVATLabels] = element(tag="labels")


class CVATMeta(BaseXmlModel, tag="meta", search_mode="unordered"):  # type: ignore
    task: Optional[CVATTask] = element(default=None)
    job: Optional[CVATJob] = element(default=None)
    project: Optional[CVATProject] = element(default=None)


class CVATBox(BaseXmlModel, tag="box", search_mode="unordered"):  # type: ignore
    label: str = attr()
    xtl: float = attr()
    ytl: float = attr()
    xbr: float = attr()
    ybr: float = attr()


class CVATImage(BaseXmlModel, tag="image", search_mode="unordered"):  # type: ignore
    id: int = attr()
    name: str = attr()  # Filename
    width: int = attr()
    height: int = attr()
    boxes: List[CVATBox] = element(tag="box", default=[])


class CVATAnnotations(BaseXmlModel, tag="annotations", search_mode="unordered"):  # type: ignore
    meta: CVATMeta = element()
    images: List[CVATImage] = element(tag="image", default=[])


class _CVATBaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to input CVAT XML file",
        )

    def __init__(self, input_file: Path) -> None:
        xml_text = input_file.read_text()
        try:
            self._data = CVATAnnotations.from_xml(xml_text)
        except Exception as ex:
            raise ValueError(f"Could not parse XML file {input_file}: {ex}") from ex

    def get_categories(self) -> Iterable["Category"]:
        meta = self._data.meta
        labels: Optional[List[CVATLabel]] = None
        if meta.task is not None and meta.task.labels:
            labels = meta.task.labels.label_list
        elif meta.job is not None and meta.job.labels:
            labels = meta.job.labels.label_list
        elif meta.project is not None and meta.project.labels:
            labels = meta.project.labels.label_list
        if labels is None:
            raise ValueError(
                "Could not find labels in meta/task, meta/job, or meta/project"
            )
        for idx, label in enumerate(labels, start=1):
            yield Category(id=idx, name=label.name)

    def get_images(self) -> Iterable["Image"]:
        for img in self._data.images:
            yield Image(
                id=img.id,
                filename=img.name,
                width=img.width,
                height=img.height,
            )


@cli_register(format="cvat", task=Task.OBJECT_DETECTION)
class CVATObjectDetectionInput(_CVATBaseInput, ObjectDetectionInput):
    def get_labels(self) -> Iterable["ImageObjectDetection"]:
        category_by_name: Dict[str, Category] = {
            cat.name: cat for cat in self.get_categories()
        }
        for img in self._data.images:
            objects = []
            for box in img.boxes:
                cat = category_by_name.get(box.label)
                if cat is None:
                    raise ParseError(f"Unknown category name '{box.label}'.")
                objects.append(
                    SingleObjectDetection(
                        category=cat,
                        box=BoundingBox.from_format(
                            bbox=[box.xtl, box.ytl, box.xbr, box.ybr],
                            format=BoundingBoxFormat.XYXY,
                        ),
                    )
                )
            yield ImageObjectDetection(
                image=Image(
                    id=img.id, filename=img.name, width=img.width, height=img.height
                ),
                objects=objects,
            )


class AnnotationScope(Enum):
    TASK = "task"
    JOB = "job"
    PROJECT = "project"

    @staticmethod
    def allowed_values() -> str:
        return ", ".join(scope.value for scope in AnnotationScope)


class _CVATBaseOutput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help="Output folder to store generated CVAT XML annotations file",
        )
        parser.add_argument(
            "--output-annotation-scope",
            choices=[scope.value for scope in AnnotationScope],
            default="task",
            help="Define the annotation scope to determine the XML structure. Allowed values: "
            + AnnotationScope.allowed_values(),
        )

    def __init__(
        self,
        output_folder: Path,
        output_annotation_scope: Literal["task", "job", "project"] = "task",
    ) -> None:
        try:
            self._annotation_scope = AnnotationScope(output_annotation_scope)
        except ValueError:
            raise ValueError(
                f"annotation_scope must be one of the allowed values: {AnnotationScope.allowed_values()}"
            )
        self._output_folder = output_folder


@cli_register(format="cvat", task=Task.OBJECT_DETECTION)
class CVATObjectDetectionOutput(_CVATBaseOutput, ObjectDetectionOutput):
    def save(self, label_input: ObjectDetectionInput) -> None:
        images = [
            CVATImage(
                id=label.image.id,
                name=label.image.filename,
                width=label.image.width,
                height=label.image.height,
                boxes=[
                    CVATBox(
                        label=obj.category.name,
                        xtl=obj.box.xmin,
                        ytl=obj.box.ymin,
                        xbr=obj.box.xmax,
                        ybr=obj.box.ymax,
                    )
                    for obj in label.objects
                ],
            )
            for label in label_input.get_labels()
        ]
        labels = CVATLabels(
            label_list=[
                CVATLabel(name=cat.name) for cat in label_input.get_categories()
            ]
        )
        if self._annotation_scope == AnnotationScope.TASK:
            meta = CVATMeta(task=CVATTask(labels=labels))
        elif self._annotation_scope == AnnotationScope.PROJECT:
            meta = CVATMeta(project=CVATProject(labels=labels))
        elif self._annotation_scope == AnnotationScope.JOB:
            meta = CVATMeta(job=CVATJob(labels=labels))
        else:
            raise ValueError(
                f"Unknown annotation_scope: {self._annotation_scope}. Allowed values: {AnnotationScope.allowed_values()}."
            )
        annotations = CVATAnnotations(meta=meta, images=images)

        self._output_folder.mkdir(parents=True, exist_ok=True)
        output_file = self._output_folder / "annotations.xml"
        # Convert the Pydantic model to an XML format (as bytes or string).
        xml_bytes = annotations.to_xml()
        # Ensure the XML output is a string — decode bytes or use as-is if it's already a string.
        if isinstance(xml_bytes, bytes):
            xml_string = xml_bytes.decode("utf-8")
        else:
            xml_string = xml_bytes
        with output_file.open("w", encoding="utf-8") as f:
            f.write(xml_string)
