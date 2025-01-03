from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv11 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov11", task=Task.OBJECT_DETECTION)
class YOLOv11ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov11", task=Task.OBJECT_DETECTION)
class YOLOv11ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
