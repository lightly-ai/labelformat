from labelformat.cli.registry import Task, cli_register

from .coco import COCOObjectDetectionInput, COCOObjectDetectionOutput

"""
RT-DETR format follows the same specs as COCO.
"""


@cli_register(format="rtdetr", task=Task.OBJECT_DETECTION)
class RTDETRObjectDetectionInput(COCOObjectDetectionInput):
    pass


@cli_register(format="rtdetr", task=Task.OBJECT_DETECTION)
class RTDETRObjectDetectionOutput(COCOObjectDetectionOutput):
    pass
