from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv6 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov6", task=Task.OBJECT_DETECTION)
class YOLOv6ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov6", task=Task.OBJECT_DETECTION)
class YOLOv6ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
