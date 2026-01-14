import json
from pathlib import Path

from labelformat.formats.youtubevis import YouTubeVISObjectDetectionTrackInput
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.object_detection_track import (
    SingleObjectDetectionTrack,
    VideoObjectDetectionTrack,
)
from labelformat.model.video import Video


class TestYouTubeVISObjectDetectionTrackInput:
    def test_get_categories(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_json(tmp_path)
        label_input = YouTubeVISObjectDetectionTrackInput(input_file=input_file)

        assert list(label_input.get_categories()) == [Category(id=1, name="cat")]

    def test_get_videos(self, tmp_path: Path) -> None:
        input_file = _write_youtube_vis_json(tmp_path)
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
        input_file = _write_youtube_vis_json(tmp_path)
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
                    )
                ],
            )
        ]

def _write_youtube_vis_json(tmp_path: Path) -> Path:
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
                "video_id": 5,
                "category_id": 1,
                "bboxes": [
                    [10.0, 20.0, 30.0, 40.0],
                    None,
                ],
            }
        ],
    }
    input_file = tmp_path / "instances.json"
    input_file.write_text(json.dumps(data))
    return input_file