from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

from labelformat.cli.registry import Task, cli_register

"""
YOLOv7 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov7", task=Task.OBJECT_DETECTION)
class YOLOv7ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov7", task=Task.OBJECT_DETECTION)
class YOLOv7ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
