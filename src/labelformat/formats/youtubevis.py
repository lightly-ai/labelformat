from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, cast

import labelformat.formats.coco_segmentation_helpers as segmentation_helpers
from labelformat.formats.coco_segmentation_helpers import (
    COCOInstanceSegmentationMultiPolygon,
    COCOInstanceSegmentationRLE,
)
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.instance_segmentation_track import (
    InstanceSegmentationTrackInput,
    SingleInstanceSegmentationTrack,
    VideoInstanceSegmentationTrack,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.object_detection_track import (
    ObjectDetectionTrackInput,
    SingleObjectDetectionTrack,
    VideoObjectDetectionTrack,
)
from labelformat.model.video import Video
from labelformat.types import JsonDict


class _YouTubeVISBaseInput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--input-file",
            type=Path,
            required=True,
            help="Path to input YouTube-VIS JSON file",
        )

    def __init__(self, input_file: Path) -> None:
        with input_file.open() as file:
            self._data = json.load(file)

    def get_categories(self) -> Iterable[Category]:
        for category in self._data["categories"]:
            yield Category(
                id=category["id"],
                name=category["name"],
            )

    def get_videos(self) -> Iterable[Video]:
        for video in self._data["videos"]:
            yield Video(
                id=video["id"],
                # TODO (Jonas, 1/2026): The file_names do not hold the video file extension. Solution required.
                filename=Path(video["file_names"][0]).parent.name,
                width=int(video["width"]),
                height=int(video["height"]),
                number_of_frames=int(video["length"]),
            )


class YouTubeVISObjectDetectionTrackInput(
    _YouTubeVISBaseInput, ObjectDetectionTrackInput
):
    def get_labels(self) -> Iterable[VideoObjectDetectionTrack]:
        video_id_to_video = {video.id: video for video in self.get_videos()}
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        video_id_to_tracks: Dict[int, List[JsonDict]] = {
            video_id: [] for video_id in video_id_to_video.keys()
        }
        for ann in self._data["annotations"]:
            video_id_to_tracks[ann["video_id"]].append(ann)

        for video_id, tracks in video_id_to_tracks.items():
            video = video_id_to_video[video_id]
            objects = []
            for track in tracks:
                boxes = _get_object_track_boxes(ann=track)
                objects.append(
                    SingleObjectDetectionTrack(
                        category=category_id_to_category[track["category_id"]],
                        boxes=boxes,
                    )
                )
            yield VideoObjectDetectionTrack(
                video=video,
                objects=objects,
            )


class YouTubeVISInstanceSegmentationTrackInput(
    _YouTubeVISBaseInput, InstanceSegmentationTrackInput
):
    def get_labels(self) -> Iterable[VideoInstanceSegmentationTrack]:
        video_id_to_video = {video.id: video for video in self.get_videos()}
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        video_id_to_tracks: Dict[int, List[JsonDict]] = {
            video_id: [] for video_id in video_id_to_video.keys()
        }
        for ann in self._data["annotations"]:
            video_id_to_tracks[ann["video_id"]].append(ann)

        for video_id, tracks in video_id_to_tracks.items():
            video = video_id_to_video[video_id]
            objects = []
            for track in tracks:
                segmentations = _get_object_track_segmentations(ann=track)
                objects.append(
                    SingleInstanceSegmentationTrack(
                        category=category_id_to_category[track["category_id"]],
                        segmentations=segmentations,
                    )
                )
            yield VideoInstanceSegmentationTrack(
                video=video,
                objects=objects,
            )


def _get_object_track_boxes(
    ann: JsonDict,
) -> list[BoundingBox | None]:
    boxes: list[BoundingBox | None] = []
    for bbox in ann["bboxes"]:
        if bbox is None or len(bbox) == 0:
            boxes.append(None)
        else:
            boxes.append(
                BoundingBox.from_format(
                    bbox=[float(x) for x in bbox],
                    format=BoundingBoxFormat.XYWH,
                )
            )
    return boxes


def _get_object_track_segmentations(
    ann: JsonDict,
) -> list[MultiPolygon | BinaryMaskSegmentation | None]:
    segmentations: list[MultiPolygon | BinaryMaskSegmentation | None] = []
    bboxes = ann["bboxes"]
    for index, segmentation in enumerate(ann["segmentations"]):
        if segmentation is None or len(segmentation) == 0:
            segmentations.append(None)
            continue
        if isinstance(segmentation, dict):
            segmentation_rle = cast(COCOInstanceSegmentationRLE, segmentation)
            segmentations.append(
                segmentation_helpers.coco_segmentation_to_binary_mask_rle(
                    segmentation=segmentation_rle, bbox=bboxes[index]
                )
            )
        elif isinstance(segmentation, list):
            segmentation_mp = cast(COCOInstanceSegmentationMultiPolygon, segmentation)
            segmentations.append(
                segmentation_helpers.coco_segmentation_to_multipolygon(
                    coco_segmentation=segmentation_mp
                )
            )
    return segmentations
