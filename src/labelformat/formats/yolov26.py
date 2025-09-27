from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv26 format follows the same specs as YOLOv11.
"""


@cli_register(format="yolov26", task=Task.OBJECT_DETECTION)
class YOLOv26ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov26", task=Task.OBJECT_DETECTION)
class YOLOv26ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
