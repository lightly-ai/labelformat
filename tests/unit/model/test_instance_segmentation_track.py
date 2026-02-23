from __future__ import annotations

import pytest

from labelformat.model.category import Category
from labelformat.model.instance_segmentation_track import (
    SingleInstanceSegmentationTrack,
    VideoInstanceSegmentationTrack,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.video import Video


class TestVideoInstanceSegmentationTrack:
    def test_post_init__frames_equal_segmentations_length__valid(self) -> None:
        track_a = SingleInstanceSegmentationTrack(
            category=Category(id=0, name="cat"),
            segmentations=[
                MultiPolygon(polygons=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]),
                None,
            ],
            object_track_id=0,
        )

        track_b = SingleInstanceSegmentationTrack(
            category=Category(id=1, name="dog"),
            segmentations=[
                MultiPolygon(polygons=[[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)]]),
                MultiPolygon(polygons=[[(4.0, 4.0), (5.0, 4.0), (5.0, 5.0)]]),
            ],
            object_track_id=1,
        )

        video = Video(id=0, filename="test.mov", width=1, height=1, number_of_frames=2)

        instance_seg = VideoInstanceSegmentationTrack(
            video=video,
            objects=[track_a, track_b],
        )
        assert len(instance_seg.objects) == 2
        assert len(instance_seg.objects[0].segmentations) == 2

    def test_post_init__frames_equal_segmentations_length___invalid(self) -> None:
        track_a = SingleInstanceSegmentationTrack(
            category=Category(id=0, name="cat"),
            segmentations=[
                MultiPolygon(polygons=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]]),
                None,
                None,
            ],
            object_track_id=0,
        )

        video = Video(id=0, filename="test.mov", width=1, height=1, number_of_frames=2)

        with pytest.raises(
            ValueError,
            match="Length of instance segmentation track does not match the number of frames in the video.",
        ):
            VideoInstanceSegmentationTrack(
                video=video,
                objects=[track_a],
            )
