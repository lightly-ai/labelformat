"""Support for image/mask pair input format for instance segmentation."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Literal, Union

from labelformat.cli.registry import Task, cli_register
from labelformat.mask_utils import (
    binarize_mask,
    extract_instance_masks,
    mask_to_binary_mask_segmentation,
    mask_to_multipolygon,
    match_image_mask_pairs,
)
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.utils import get_image_dimensions


@cli_register(format="maskpair", task=Task.INSTANCE_SEGMENTATION)
class MaskPairInstanceSegmentationInput(InstanceSegmentationInput):
    """Input format for image/mask pairs for instance segmentation.

    This format loads images and corresponding binary masks, converts masks to
    instance segmentations by finding connected components, and outputs in the
    standard labelformat representation.
    """

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--image-glob",
            type=str,
            required=True,
            help="Glob pattern for image files (e.g., 'images/**/*.jpg')",
        )
        parser.add_argument(
            "--mask-glob",
            type=str,
            required=True,
            help="Glob pattern for mask files (e.g., 'masks/**/*.png')",
        )
        parser.add_argument(
            "--base-path",
            type=Path,
            default=Path("."),
            help="Base directory for glob patterns (default: current directory)",
        )
        parser.add_argument(
            "--pairing-mode",
            choices=["stem", "regex", "index"],
            default="stem",
            help="How to match images to masks: stem (filename), regex (numeric ID), index (sorted order)",
        )
        parser.add_argument(
            "--category-names",
            type=str,
            required=True,
            help="Comma-separated category names (e.g., 'crack,defect')",
        )
        parser.add_argument(
            "--threshold",
            type=int,
            default=-1,
            help="Binarization threshold 0-255, negative for automatic Otsu",
        )
        parser.add_argument(
            "--min-area",
            type=float,
            default=10.0,
            help="Minimum instance area in pixels to include",
        )
        parser.add_argument(
            "--morph-open",
            type=int,
            default=0,
            help="Morphological opening kernel size in pixels (0 to disable)",
        )
        parser.add_argument(
            "--morph-close",
            type=int,
            default=0,
            help="Morphological closing kernel size in pixels (0 to disable)",
        )
        parser.add_argument(
            "--segmentation-type",
            choices=["polygon", "rle"],
            default="polygon",
            help="Output segmentation format: polygon or RLE",
        )
        parser.add_argument(
            "--approx-epsilon",
            type=float,
            default=0.0,
            help="Polygon approximation factor (0.0 for full detail)",
        )

    def __init__(
        self,
        image_glob: str,
        mask_glob: str,
        base_path: Path = Path("."),
        pairing_mode: Literal["stem", "regex", "index"] = "stem",
        category_names: str = "object",
        threshold: int = -1,
        min_area: float = 10.0,
        morph_open: int = 0,
        morph_close: int = 0,
        segmentation_type: Literal["polygon", "rle"] = "polygon",
        approx_epsilon: float = 0.0,
    ) -> None:
        self._image_glob = image_glob
        self._mask_glob = mask_glob
        self._base_path = Path(base_path)
        self._pairing_mode = pairing_mode
        self._threshold = None if threshold < 0 else threshold
        self._min_area = min_area
        self._morph_open = morph_open
        self._morph_close = morph_close
        self._segmentation_type = segmentation_type
        self._approx_epsilon = approx_epsilon

        # Parse category names
        self._categories = []
        for i, name in enumerate(category_names.split(",")):
            self._categories.append(Category(id=i, name=name.strip()))

        # Find and validate image/mask pairs
        self._image_mask_pairs = match_image_mask_pairs(
            image_glob=image_glob,
            mask_glob=mask_glob,
            base_path=self._base_path,
            pairing_mode=pairing_mode,
        )

    def get_categories(self) -> Iterable[Category]:
        """Get the categories for this dataset."""
        yield from self._categories

    def get_images(self) -> Iterable[Image]:
        """Get images from the image/mask pairs."""
        for image_id, (image_path, _) in enumerate(self._image_mask_pairs):
            width, height = get_image_dimensions(image_path)
            yield Image(
                id=image_id,
                filename=image_path.name,
                width=width,
                height=height,
            )

    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        """Get instance segmentation labels by processing mask images."""
        images = {img.id: img for img in self.get_images()}

        for image_id, (image_path, mask_path) in enumerate(self._image_mask_pairs):
            # Load and binarize the mask
            binary_mask = binarize_mask(
                mask_path=mask_path,
                threshold=self._threshold,
                morph_open=self._morph_open,
                morph_close=self._morph_close,
            )

            # Extract individual instances via connected components
            instance_masks = extract_instance_masks(binary_mask)

            # Convert each instance to segmentation format
            objects: List[SingleInstanceSegmentation] = []
            for instance_mask in instance_masks:
                # Skip instances that are too small
                area = float(instance_mask.sum())
                if area < self._min_area:
                    continue

                # Convert to appropriate segmentation format
                segmentation: Union[MultiPolygon, BinaryMaskSegmentation]
                if self._segmentation_type == "polygon":
                    segmentation = mask_to_multipolygon(
                        binary_mask=instance_mask,
                        approx_epsilon=self._approx_epsilon,
                    )
                elif self._segmentation_type == "rle":
                    segmentation = mask_to_binary_mask_segmentation(instance_mask)
                else:
                    raise ValueError(
                        f"Invalid segmentation type: {self._segmentation_type}."
                        "Valid options are: polygon, rle"
                    )

                # Use the first category for now (could be extended to support multiple categories)
                category = (
                    self._categories[0]
                    if self._categories
                    else Category(id=0, name="object")
                )

                objects.append(
                    SingleInstanceSegmentation(
                        category=category,
                        segmentation=segmentation,
                    )
                )

            yield ImageInstanceSegmentation(
                image=images[image_id],
                objects=objects,
            )
