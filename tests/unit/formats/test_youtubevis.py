import json
from pathlib import Path

import numpy as np

from labelformat.formats.youtubevis import (
    YouTubeVISInstanceSegmentationTrackInput,
    YouTubeVISInstanceSegmentationTrackOutput,
    YouTubeVISObjectDetectionTrackInput,
)
from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.instance_segmentation_track import (
    SingleInstanceSegmentationTrack,
    VideoInstanceSegmentationTrack,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.object_detection_track import (
    SingleObjectDetectionTrack,
    VideoObjectDetectionTrack,
)
from labelformat.model.video import Video


class TestYouTubeVISObjectDetectionTrackInput:
    def test_get_categories(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_json(tmp_path / "instances.json")
        label_input = YouTubeVISObjectDetectionTrackInput(input_file=input_file)

        assert list(label_input.get_categories()) == [Category(id=1, name="cat")]

    def test_get_videos(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_json(tmp_path / "instances.json")
        label_input = YouTubeVISObjectDetectionTrackInput(input_file=input_file)

        assert list(label_input.get_videos()) == [
            Video(
                id=5,
                filename="video1",
                width=640,
                height=480,
                number_of_frames=2,
            )
        ]

    def test_get_labels(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_json(tmp_path / "instances.json")
        label_input = YouTubeVISObjectDetectionTrackInput(input_file=input_file)

        assert list(label_input.get_labels()) == [
            VideoObjectDetectionTrack(
                video=Video(
                    id=5,
                    filename="video1",
                    width=640,
                    height=480,
                    number_of_frames=2,
                ),
                objects=[
                    SingleObjectDetectionTrack(
                        category=Category(id=1, name="cat"),
                        boxes=[
                            BoundingBox(
                                xmin=10.0,
                                ymin=20.0,
                                xmax=40.0,
                                ymax=60.0,
                            ),
                            None,
                        ],
                        object_track_id=10,
                    )
                ],
            )
        ]


class TestYouTubeVISInstanceSegmentationTrackInput:
    def test_get_labels(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_instance_segmentation_json(
            tmp_path / "instances.json"
        )
        label_input = YouTubeVISInstanceSegmentationTrackInput(input_file=input_file)

        binary_mask = np.array([[0, 1, 1], [0, 1, 1]], dtype=int)
        bounding_box = BoundingBox(xmin=1.0, ymin=0.0, xmax=3.0, ymax=2.0)
        expected_rle_segmentation = BinaryMaskSegmentation.from_binary_mask(
            binary_mask=binary_mask,
            bounding_box=bounding_box,
        )

        expected_polygon_segmentation = MultiPolygon(
            polygons=[
                [
                    (10.0, 10.0),
                    (20.0, 10.0),
                    (20.0, 20.0),
                    (10.0, 20.0),
                ]
            ]
        )

        assert list(label_input.get_labels()) == [
            VideoInstanceSegmentationTrack(
                video=Video(
                    id=5,
                    filename="video1",
                    width=3,
                    height=2,
                    number_of_frames=2,
                ),
                objects=[
                    SingleInstanceSegmentationTrack(
                        category=Category(id=1, name="cat"),
                        segmentations=[expected_rle_segmentation, None],
                        object_track_id=20,
                    ),
                    SingleInstanceSegmentationTrack(
                        category=Category(id=1, name="cat"),
                        segmentations=[
                            expected_polygon_segmentation,
                            None,
                        ],
                        object_track_id=21,
                    ),
                ],
            )
        ]


class TestYouTubeVISInstanceSegmentationTrackInputExportImport:
    def test_import_export(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_instance_segmentation_json(
            tmp_path / "instances.json"
        )
        label_input = YouTubeVISInstanceSegmentationTrackInput(input_file=input_file)

        output_path = tmp_path / "instances_out.json"
        YouTubeVISInstanceSegmentationTrackOutput(output_file=output_path).save(
            label_input=label_input
        )
        output_json = json.loads(output_path.read_text())
        expected_json = json.loads(input_file.read_text())
        cat_output = output_json["categories"]
        cat_expected = expected_json["categories"]
        assert (
            cat_output == cat_expected
        ), f"Categories do not match. Output: {cat_output}, Expected: {cat_expected}"
        video_output = output_json["videos"]
        video_expected = expected_json["videos"]
        assert (
            video_output == video_expected
        ), f"Videos do not match. Output: {video_output}, Expected: {video_expected}"
        annotation_output = output_json["annotations"]
        annotation_expected = expected_json["annotations"]
        assert (
            annotation_output == annotation_expected
        ), f"Annotations do not match. Output: {annotation_output}, Expected: {annotation_expected}"


def _write_youtube_vis_json(input_file: Path) -> Path:
    data = {
        "categories": [
            {"id": 1, "name": "cat"},
        ],
        "videos": [
            {
                "id": 5,
                "file_names": ["video1/00000.jpg", "video1/00001.jpg"],
                "width": 640,
                "height": 480,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 10,
                "video_id": 5,
                "category_id": 1,
                "bboxes": [
                    [10.0, 20.0, 30.0, 40.0],
                    None,
                ],
            }
        ],
    }
    input_file.write_text(json.dumps(data))
    return input_file


def _write_youtube_vis_instance_segmentation_json(input_file: Path) -> Path:
    binary_mask = np.array([[0, 1, 1], [0, 1, 1]], dtype=int)
    counts = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
    data = {
        "categories": [
            {"id": 1, "name": "cat"},
        ],
        "videos": [
            {
                "id": 5,
                "file_names": ["video1/00000.jpg", "video1/00001.jpg"],
                "width": 3,
                "height": 2,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 20,
                "video_id": 5,
                "category_id": 1,
                "iscrowd": 1,
                "height": 2,
                "width": 3,
                "length": 2,
                "bboxes": [
                    [1.0, 0.0, 2.0, 2.0],
                    None,
                ],
                "segmentations": [
                    {"counts": counts, "size": [2, 3]},
                    None,
                ],
                "areas": [4, None],
            },
            {
                "id": 21,
                "video_id": 5,
                "category_id": 1,
                "iscrowd": 0,
                "height": 2,
                "width": 3,
                "length": 2,
                "bboxes": [
                    [10.0, 10.0, 10.0, 10.0],
                    None,
                ],
                "segmentations": [
                    [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
                    None,
                ],
                # Currently we do not fill the area for multypolygon segmentations.
                "areas": [None, None],
            },
        ],
    }
    input_file.write_text(json.dumps(data))
    return input_file
