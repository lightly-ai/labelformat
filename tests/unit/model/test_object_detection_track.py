from __future__ import annotations

import pytest

from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.object_detection_track import (
    SingleObjectDetectionTrack,
    VideoObjectDetectionTrack,
)
from labelformat.model.video import Video


class TestVideoObjectDetectionTrack:
    def test_frames_equal_boxes_length__valid(self) -> None:
        track_a = SingleObjectDetectionTrack(
            category=Category(id=0, name="cat"),
            boxes=[BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1) for _ in range(2)],
        )

        track_b = SingleObjectDetectionTrack(
            category=Category(id=1, name="dog"),
            boxes=[BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1) for _ in range(2)],
        )

        video = Video(id=0, filename="test.mov", width=1, height=1, number_of_frames=2)

        detections = VideoObjectDetectionTrack(
            video=video,
            objects=[track_a, track_b],
        )
        assert len(detections.objects) == 2
        assert len(detections.objects[0].boxes) == 2

    def test_frames_equal_boxes_length___invalid(self) -> None:
        track_a = SingleObjectDetectionTrack(
            category=Category(id=0, name="cat"),
            boxes=[BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1) for _ in range(2)],
        )

        track_b = SingleObjectDetectionTrack(
            category=Category(id=1, name="dog"),
            boxes=[BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1) for _ in range(3)],
        )

        video = Video(id=0, filename="test.mov", width=1, height=1, number_of_frames=2)

        with pytest.raises(
            ValueError,
            match="Length of object detection track does not match the number of frames in the video.",
        ):
            VideoObjectDetectionTrack(
                video=video,
                objects=[track_a, track_b],
            )
