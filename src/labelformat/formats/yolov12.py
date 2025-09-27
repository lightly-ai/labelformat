from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv12 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov12", task=Task.OBJECT_DETECTION)
class YOLOv12ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov12", task=Task.OBJECT_DETECTION)
class YOLOv12ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
