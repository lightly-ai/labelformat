from labelformat.cli.registry import Task, cli_register

from .yolov8 import YOLOv8ObjectDetectionInput, YOLOv8ObjectDetectionOutput

"""
YOLOv10 format follows the same specs as YOLOv8.
"""


@cli_register(format="yolov10", task=Task.OBJECT_DETECTION)
class YOLOv10ObjectDetectionInput(YOLOv8ObjectDetectionInput):
    pass


@cli_register(format="yolov10", task=Task.OBJECT_DETECTION)
class YOLOv10ObjectDetectionOutput(YOLOv8ObjectDetectionOutput):
    pass
