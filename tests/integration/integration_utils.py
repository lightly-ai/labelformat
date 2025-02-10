from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from labelformat.model.multipolygon import MultiPolygon, Point

INST_SEGMENTATION_FIXTURES_DIR = (
    Path(__file__).parent.parent / "fixtures/instance_segmentation"
)

OBJ_DETECTION_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures/object_detection"

VIDEO_INSTANCE_SEGMENTATION_FIXTURES_DIR = (
    Path(__file__).parent.parent / "fixtures/video_instance_segmentation"
)

COMMA_JOINED_CATEGORY_NAMES = ",".join(
    [
        "person",
        "bicycle with space",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
    ]
)


def assert_almost_equal_recursive(
    obj1: object,
    obj2: object,
    rel: Optional[float] = None,
    abs: Optional[float] = None,
    nan_ok: bool = False,
) -> None:
    if isinstance(obj1, dict):
        if 'counts' in obj1: #For RLE encodded segmentations
            import pycocotools.mask as mask_utils
            mask1 = mask_utils.decode(obj1)
            mask2 = mask_utils.decode(obj2)
            assert mask1.shape == mask2.shape, "RLE masks have different shapes"
            # Allow for subtle differences by using a tolerance
            difference = np.abs(mask1 - mask2).sum()
            tolerance = 5  # Adjust tolerance as needed
            assert (difference <= tolerance), "RLE masks differ beyond tolerance"
        else:
            assert isinstance(obj2, dict)
            assert sorted(obj1.keys()) == sorted(obj2.keys())
            for key in obj1.keys():
                assert_almost_equal_recursive(
                    obj1[key], obj2[key], rel=rel, abs=abs, nan_ok=nan_ok
                )
    elif isinstance(obj1, list):
        assert isinstance(obj2, list)
        assert len(obj1) == len(obj2)
        for item1, item2 in zip(obj1, obj2):
            assert_almost_equal_recursive(item1, item2, rel=rel, abs=abs, nan_ok=nan_ok)
    elif isinstance(obj1, float) or isinstance(obj1, int):
        assert isinstance(obj2, float) or isinstance(obj2, int)
        assert pytest.approx(float(obj1), rel=1e-1) == float(obj2)
    else:
        assert obj1 == obj2


def assert_multipolygons_almost_equal(a: MultiPolygon, b: MultiPolygon) -> None:
    """
    Heuristic test that two MultiPolygons cover the same area.

    Ideally we would compute the intersection and union of the two MultiPolygon,
    which is non-trivial without a helper library. Instead we just check that
    * The set of points is almost equal
    * Their areas are almost equal
    """
    precision = 3
    points_a = {
        (round(p[0], ndigits=precision), round(p[1], ndigits=precision))
        for polygon in a.polygons
        for p in polygon
    }
    points_b = {
        (round(p[0], ndigits=precision), round(p[1], ndigits=precision))
        for polygon in b.polygons
        for p in polygon
    }
    assert points_a == points_b, "multipolygons consist of a different set of points"

    area_a = sum(_polygon_area(polygon) for polygon in a.polygons)
    area_b = sum(_polygon_area(polygon) for polygon in b.polygons)
    assert abs(area_a - area_b) < (10 ** (-precision)), "multipolygon areas differ"


def _polygon_area(polygon: List[Point]) -> float:
    """Compute the area of a polygon."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    area = abs(area) / 2.0
    return area
