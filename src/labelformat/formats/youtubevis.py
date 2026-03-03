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
from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.instance_segmentation_track import (
    InstanceSegmentationTrackInput,
    InstanceSegmentationTrackOutput,
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
                        object_track_id=track.get("id"),
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
                        object_track_id=track.get("id"),
                    )
                )
            yield VideoInstanceSegmentationTrack(
                video=video,
                objects=objects,
            )


class _YouTubeVISBaseOutput:
    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-file",
            type=Path,
            required=True,
            help="Path to output YouTube-VIS JSON file",
        )

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file


class YouTubeVISInstanceSegmentationTrackOutput(
    _YouTubeVISBaseOutput, InstanceSegmentationTrackOutput
):
    def save(self, label_input: InstanceSegmentationTrackInput) -> None:
        data: JsonDict = {
            "info": {"description": "YouTube-VIS export"},
            "videos": _get_output_videos_dict(label_input.get_videos()),
            "categories": _get_output_categories_dict(label_input.get_categories()),
            "annotations": _get_output_annotations_dict(label_input.get_labels()),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w") as f:
            json.dump(data, f, indent=2)



def _get_output_videos_dict(videos: Iterable[Video]) -> List[JsonDict]:
    """Get the 'videos' list for YouTube-VIS JSON."""
    result = []
    for video in videos:
        file_names = [
            f"{video.filename}/{i:05d}.jpg" for i in range(video.number_of_frames)
        ]
        result.append(
            {
                "id": video.id,
                "file_names": file_names,
                "width": video.width,
                "height": video.height,
                "length": video.number_of_frames,
            }
        )
    return result


def _get_output_categories_dict(
    categories: Iterable[Category],
) -> List[JsonDict]:
    """Get the 'categories' list for YouTube-VIS JSON."""
    return [{"id": category.id, "name": category.name} for category in categories]


def _get_output_annotations_dict(
    labels: Iterable[VideoInstanceSegmentationTrack],
) -> List[JsonDict]:
    """Get the 'annotations' list for YouTube-VIS JSON."""
    result: List[JsonDict] = []
    for label in labels:
        video = label.video
        length = video.number_of_frames
        height, width = video.height, video.width
        for obj in label.objects:
            bboxes_list: List[list[float] | None] = []
            segmentations_list: List[
                COCOInstanceSegmentationRLE
                | COCOInstanceSegmentationMultiPolygon
                | None
            ] = []
            areas_list: List[int | None] = []
            for seg in obj.segmentations:
                if seg is None:
                    bboxes_list.append(None)
                    segmentations_list.append(None)
                    areas_list.append(None)
                    continue
                out_seg = _segmentation_to_youtube_vis(seg, None)
                if isinstance(seg, BinaryMaskSegmentation):
                    bbox_xywh = [
                        float(v)
                        for v in seg.bounding_box.to_format(BoundingBoxFormat.XYWH)
                    ]
                    col_rle = RLEDecoderEncoder.encode_column_wise_rle(
                        seg.get_binary_mask()
                    )
                    area = sum(col_rle[1::2])  # foreground runs at odd indices
                elif isinstance(seg, MultiPolygon):
                    bbox = seg.bounding_box()
                    bbox_xywh = [
                        float(v) for v in bbox.to_format(BoundingBoxFormat.XYWH)
                    ]
                    area = int(bbox_xywh[2] * bbox_xywh[3])
                else:
                    bbox_xywh = None
                    area = None
                bboxes_list.append(bbox_xywh)
                segmentations_list.append(out_seg)
                areas_list.append(area)
            result.append(
                {
                    "id": obj.object_track_id,
                    "video_id": video.id,
                    "category_id": obj.category.id,
                    "bboxes": bboxes_list,
                    "segmentations": segmentations_list,
                    "areas": areas_list,
                    "iscrowd": 0,
                    "height": height,
                    "width": width,
                    "length": length,
                }
            )
    return result


def _segmentation_to_youtube_vis(
    segmentation: MultiPolygon | BinaryMaskSegmentation | None,
    bbox_xywh: list[float] | None,
) -> COCOInstanceSegmentationRLE | COCOInstanceSegmentationMultiPolygon | None:
    """Convert a single-frame segmentation to YouTube-VIS format (RLE or polygon list)."""
    if segmentation is None:
        return None
    if isinstance(segmentation, BinaryMaskSegmentation):
        binary_mask = segmentation.get_binary_mask()
        counts = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
        return {"counts": counts, "size": [segmentation.height, segmentation.width]}
    if isinstance(segmentation, MultiPolygon):
        coco_seg = []
        for polygon in segmentation.polygons:
            coco_seg.append([x for point in polygon for x in point])
        return coco_seg
    return None


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
