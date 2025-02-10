from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Type

from labelformat.model.instance_segmentation import (
    InstanceSegmentationInput,
    InstanceSegmentationOutput,
)
from labelformat.model.object_detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
)
from labelformat.model.video_instance_segmentation import (
    VideoInstanceSegmentationInput,
    VideoInstanceSegmentationOutput,
)


class Task(Enum):
    INSTANCE_SEGMENTATION = "instance-segmentation"
    OBJECT_DETECTION = "object-detection"
    VIDEO_INSTANCE_SEGMENTATION = "video-instance-segmentation"


@dataclass
class Registry:
    input: Dict[Task, Dict[str, Type]]  # type: ignore[type-arg]
    output: Dict[Task, Dict[str, Type]]  # type: ignore[type-arg]


_REGISTRY = Registry(
    input={task: {} for task in Task}, output={task: {} for task in Task}
)


def cli_register(format: str, task: Task) -> Callable[[Type], Type]:  # type: ignore[type-arg]
    def decorator(cls: Type) -> Type:  # type: ignore[type-arg]
        if (
            issubclass(cls, ObjectDetectionInput)
            or issubclass(cls, InstanceSegmentationInput)
            or issubclass(cls, VideoInstanceSegmentationInput)
        ):
            _REGISTRY.input[task][format] = cls
        elif (
            issubclass(cls, ObjectDetectionOutput)
            or issubclass(cls, InstanceSegmentationOutput)
            or issubclass(cls, VideoInstanceSegmentationOutput)
        ):
            _REGISTRY.output[task][format] = cls
        else:
            raise ValueError(
                "Can only register classes which extend one of: "
                f"'{ObjectDetectionInput}', "
                f"'{InstanceSegmentationInput}', "
                f"'{ObjectDetectionOutput}', "
                f"'{InstanceSegmentationOutput}', "
                f"'{VideoInstanceSegmentationInput}', "
                f"'{VideoInstanceSegmentationOutput}'"
            )
        return cls

    return decorator
