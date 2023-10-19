from dataclasses import dataclass
from enum import Enum
from typing import List

from labelformat.model.category import Category


class BoundingBoxFormat(Enum):
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"


@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_format(
        bbox: List[float],
        format: BoundingBoxFormat,
    ) -> "BoundingBox":
        """Create a bounding box from a list of floats and a format.

        We assume all bounding box coordinates are in pixel coordinates and are
        NOT normalized between 0 and 1.

        Args:
            bbox (List[float]): A list of floats representing the bounding box.
            format (BoundingBoxFormat): The format of the bounding box.
        """
        if format == BoundingBoxFormat.XYXY:
            return BoundingBox(
                xmin=bbox[0],
                ymin=bbox[1],
                xmax=bbox[2],
                ymax=bbox[3],
            )
        elif format == BoundingBoxFormat.XYWH:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            return BoundingBox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        elif format == BoundingBoxFormat.CXCYWH:
            xmin = bbox[0] - bbox[2] / 2
            ymin = bbox[1] - bbox[3] / 2
            xmax = bbox[0] + bbox[2] / 2
            ymax = bbox[1] + bbox[3] / 2
            return BoundingBox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        else:
            raise ValueError(
                f"Unknown bbox format: {format}, known formats are {list(BoundingBoxFormat)}"
            )

    def to_format(self, format: BoundingBoxFormat) -> List[float]:
        if format == BoundingBoxFormat.XYXY:
            return [self.xmin, self.ymin, self.xmax, self.ymax]
        elif format == BoundingBoxFormat.XYWH:
            return [self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin]
        elif format == BoundingBoxFormat.CXCYWH:
            return [
                (self.xmin + self.xmax) / 2,
                (self.ymin + self.ymax) / 2,
                self.xmax - self.xmin,
                self.ymax - self.ymin,
            ]
        else:
            raise ValueError(
                f"Unknown bbox format: {format}, known formats are {list(BoundingBoxFormat)}"
            )
