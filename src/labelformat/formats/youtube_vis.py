import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np
import pycocotools.mask as mask_utils

from labelformat.cli.registry import Task, cli_register
from labelformat.model.category import Category
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.video import Video
from labelformat.model.video_instance_segmentation import (
    SingleVideoInstanceSegmentation,
    VideoInstanceSegmentation,
    VideoInstanceSegmentationInput,
    VideoInstanceSegmentationOutput,
)
from labelformat.types import JsonDict, ParseError


@cli_register(format="youtube_vis", task=Task.VIDEO_INSTANCE_SEGMENTATION)
class YouTubeVISInput(VideoInstanceSegmentationInput):
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
                filenames=video["file_names"],
                width=int(video["width"]),
                height=int(video["height"]),
                length=int(video["length"]),
            )

    def get_labels(self) -> Iterable[VideoInstanceSegmentation]:
        video_id_to_video = {video.id: video for video in self.get_videos()}
        category_id_to_category = {
            category.id: category for category in self.get_categories()
        }
        video_id_to_annotations: Dict[int, List[JsonDict]] = {
            video_id: [] for video_id in video_id_to_video.keys()
        }
        for ann in self._data["annotations"]:
            video_id_to_annotations[ann["video_id"]].append(ann)

        for video_id, annotations in video_id_to_annotations.items():
            objects = []
            for ann in annotations:
                if "segmentations" not in ann:
                    raise ParseError(f"Segmentations missing for video id {video_id}")
                segmentations = _youtube_vis_segmentation_to_multipolygon(ann["segmentations"])
                objects.append(
                    SingleVideoInstanceSegmentation(
                        category=category_id_to_category[ann["category_id"]],
                        segmentation=segmentations,
                    )
                )
            yield VideoInstanceSegmentation(
                video=video_id_to_video[video_id],
                objects=objects,
            )


@cli_register(format="youtube_vis", task=Task.VIDEO_INSTANCE_SEGMENTATION)
class YouTubeVISOutput(VideoInstanceSegmentationOutput):
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-file",
            type=Path,
            required=True,
            help="Path to output YouTube-VIS JSON file",
        )

    def save(self, label_input: VideoInstanceSegmentationInput) -> None:
        data = {}
        data["videos"] = _get_output_videos_dict(videos=label_input.get_videos())
        data["categories"] = _get_output_categories_dict(
            categories=label_input.get_categories()
        )
        data["annotations"] = []
        unique_id = 1  # Initialize a counter for unique IDs
        for label in label_input.get_labels():
            for id, obj in enumerate(label.objects):
                annotation = {
                    "video_id": label.video.id,
                    "category_id": obj.category.id,
                    "segmentations": _multipolygon_to_youtube_vis_segmentation(obj.segmentation,
                                                                                label.video.height,
                                                                                label.video.width),
                    "id": unique_id,
                    "width": label.video.width,
                    "height": label.video.height,
                    "iscrowd": 0,                    
                    "occlusion": ['no_occlusion' for _ in range(label.video.length)],
                }
                data["annotations"].append(annotation)
                unique_id += 1

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open  ("w") as file:
            json.dump(data, file, indent=2)

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file



def _youtube_vis_segmentation_to_multipolygon(
    youtube_vis_segmentation: List[Union[List[float], Dict[str, Any]]],
) -> MultiPolygon:
    """Convert YouTube-VIS segmentation to MultiPolygon."""
    polygons = []
    for polygon in youtube_vis_segmentation:
        if isinstance(polygon, dict) and "counts" in polygon and "size" in polygon:
            # Convert RLE format to polygon
            binary_mask = mask_utils.decode(polygon)
            contours = _mask_to_polygons(binary_mask)
            # Process each contour the same way as regular polygons
            for contour in contours:
                polygons.append(
                    list(
                        zip(
                            [float(x) for x in contour[:, 0]],
                            [float(x) for x in contour[:, 1]],
                        )
                    )
                )
        else:
            # Handle polygon format
            if len(polygon) % 2 != 0:
                raise ParseError(
                    f"Invalid polygon with {len(polygon)} points: {polygon}"
                )
            polygons.append(
                list(
                    zip(
                        [float(x) for x in polygon[0::2]],
                        [float(x) for x in polygon[1::2]],
                    )
                )
            )
    return MultiPolygon(polygons=polygons)


def _mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    """Convert binary mask to list of contours."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [contour.squeeze() for contour in contours if len(contour) >= 3]


def _multipolygon_to_youtube_vis_segmentation(
    multipolygon: MultiPolygon,
    height: int,
    width: int,
) -> List[Union[List[float], Dict[str, Any]]]:
    """Convert MultiPolygon to YouTube-VIS segmentation."""
    youtube_vis_segmentation = []
    for polygon in multipolygon.polygons:
        # Convert polygon to RLE format
        mask = np.zeros((height, width), dtype=np.uint8)  # Define the mask size
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')  # Ensure counts is a string
        youtube_vis_segmentation.append(rle)
    return youtube_vis_segmentation

def _get_output_videos_dict(
    videos: Iterable[Video],
) -> List[JsonDict]:
    """Get the "videos" dict for YouTube-VIS JSON."""
    return [
        {
            "id": video.id,
            "file_names": video.filenames,
            "length": video.length,
            "width": video.width,
            "height": video.height,
        }
        for video in videos
    ]


def _get_output_categories_dict(
    categories: Iterable[Category],
) -> List[JsonDict]:
    """Get the "categories" dict for YouTube-VIS JSON."""
    return [
        {
            "id": category.id,
            "name": category.name,
        }
        for category in categories
    ]
