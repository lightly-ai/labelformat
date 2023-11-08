from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv5 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov5", task=Task.OBJECT_DETECTION)
class YOLOv5ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov5", task=Task.OBJECT_DETECTION)
class YOLOv5ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
