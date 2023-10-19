from dataclasses import dataclass
from typing import List, Tuple

from labelformat.model.bounding_box import BoundingBox

Point = Tuple[float, float]


@dataclass(frozen=True)
class MultiPolygon:
    """MultiPolygon for instance segmentation.

    We assume all bounding box coordinates are in pixel coordinates and are
    NOT normalized between 0 and 1.
    """

    polygons: List[List[Point]]

    def bounding_box(self) -> BoundingBox:
        """Get the bounding box of this MultiPolygon."""
        if len(self.polygons) == 0:
            raise ValueError("Cannot get bounding box of empty MultiPolygon.")

        xmin = self.polygons[0][0][0]
        ymin = self.polygons[0][0][1]
        xmax = self.polygons[0][0][0]
        ymax = self.polygons[0][0][1]

        for polygon in self.polygons:
            for point in polygon:
                xmin = min(xmin, point[0])
                ymin = min(ymin, point[1])
                xmax = max(xmax, point[0])
                ymax = max(ymax, point[1])

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
