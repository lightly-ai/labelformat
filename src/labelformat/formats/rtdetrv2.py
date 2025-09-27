from labelformat.cli.registry import Task, cli_register

from .coco import COCOObjectDetectionInput, COCOObjectDetectionOutput

"""
RT-DETRv2 format follows the same specs as COCO.
"""


@cli_register(format="rtdetrv2", task=Task.OBJECT_DETECTION)
class RTDETRv2ObjectDetectionInput(COCOObjectDetectionInput):
    pass


@cli_register(format="rtdetrv2", task=Task.OBJECT_DETECTION)
class RTDETRv2ObjectDetectionOutput(COCOObjectDetectionOutput):
    pass
