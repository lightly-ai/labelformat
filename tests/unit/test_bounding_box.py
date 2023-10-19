from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category


class TestBoundingBox:
    def test_bounding_box(self) -> None:
        bounding_box_base = BoundingBox(
            xmin=10.0,
            ymin=20.0,
            xmax=30.0,
            ymax=40.0,
        )

        bounding_box_yolo = bounding_box_base.from_format(
            bbox=[10.0, 20.0, 20.0, 20.0],
            format=BoundingBoxFormat.XYWH,
        )

        assert bounding_box_yolo.xmin == bounding_box_base.xmin
        assert bounding_box_yolo.ymin == bounding_box_base.ymin
        assert bounding_box_yolo.xmax == bounding_box_base.xmax
        assert bounding_box_yolo.ymax == bounding_box_base.ymax

    def test_bounding_box_conversions(self) -> None:
        bounding_box_base = BoundingBox(
            xmin=10.0,
            ymin=20.0,
            xmax=30.0,
            ymax=40.0,
        )

        bounding_box_xywh = bounding_box_base.to_format(BoundingBoxFormat.XYWH)
        assert bounding_box_xywh == [10.0, 20.0, 20.0, 20.0]

        bounding_box_xyxy = bounding_box_base.to_format(BoundingBoxFormat.XYXY)
        assert bounding_box_xyxy == [10.0, 20.0, 30.0, 40.0]

        bounding_box_cxcywh = bounding_box_base.to_format(BoundingBoxFormat.CXCYWH)
        assert bounding_box_cxcywh == [20.0, 30.0, 20.0, 20.0]
