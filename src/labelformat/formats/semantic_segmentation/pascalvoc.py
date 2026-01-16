"""Pascal VOC semantic segmentation input.

Assumptions:
- Masks live under a separate directory mirroring the images directory structure.
- For each image at ``images_dir/<rel>.ext``, the mask is at ``masks_dir/<rel>.png``.
- Masks are PNGs with pixel values equal to class IDs.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from labelformat import utils
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    SingleInstanceSegmentation,
)
from labelformat.model.semantic_segmentation import SemanticSegmentationMask

"""TODO(Malte, 11/2025):
Support what is already supported in LightlyTrain. https://docs.lightly.ai/train/stable/semantic_segmentation.html#data
Support using a template against the image filepath. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-a-template-against-the-image-filepath
Support using multi-channel masks. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-multi-channel-masks
Support optional ignore_classes: list[int] that should be ignored during training. https://docs.lightly.ai/train/stable/semantic_segmentation.html#specify-training-classes
Support merging multiple labels into one class during training. https://docs.lightly.ai/train/stable/semantic_segmentation.html#specify-training-classes
"""


@dataclass
class PascalVOCSemanticSegmentationInput(InstanceSegmentationInput):
    """Pascal VOC semantic segmentation input format."""

    _images_dir: Path
    _masks_dir: Path
    _filename_to_image: dict[str, Image]
    _categories: list[Category]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        # TODO(Michal, 01/2026): Implement when needed.
        raise NotImplementedError()

    @classmethod
    def from_dirs(
        cls,
        images_dir: Path,
        masks_dir: Path,
        class_id_to_name: Mapping[int, str],
    ) -> "PascalVOCSemanticSegmentationInput":
        """Create a PascalVOCSemanticSegmentationInput from directory pairs.

        Args:
            images_dir: Root directory containing images (nested structure allowed).
            masks_dir: Root directory containing PNG masks mirroring images structure.
            class_id_to_name: Mapping of class_id -> class name, with integer keys.

        Raises:
            ValueError: If directories are invalid, a mask is missing or not PNG,
                        or if class_id keys cannot be parsed as integers.
        """
        if not images_dir.is_dir():
            raise ValueError(f"Images directory is not a directory: {images_dir}")
        if not masks_dir.is_dir():
            raise ValueError(f"Masks directory is not a directory: {masks_dir}")

        # Build categories from mapping
        categories = [
            Category(id=cid, name=cname) for cid, cname in class_id_to_name.items()
        ]

        # Collect images using helper and ensure a PNG mask exists for each.
        images_by_filename: dict[str, Image] = {}
        for img in utils.get_images_from_folder(images_dir):
            mask_path = masks_dir / Path(img.filename).with_suffix(".png")
            if not mask_path.is_file():
                raise ValueError(
                    f"Missing mask PNG for image '{img.filename}' at path: {mask_path}"
                )
            images_by_filename[img.filename] = img

        return cls(
            _images_dir=images_dir,
            _masks_dir=masks_dir,
            _filename_to_image=images_by_filename,
            _categories=categories,
        )

    def get_categories(self) -> Iterable[Category]:
        return list(self._categories)

    def get_images(self) -> Iterable[Image]:
        yield from self._filename_to_image.values()

    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        """Get semantic segmentation labels.

        Yields an object per image, with one binary mask per category present in the mask.
        The order of objects is sorted by category ID. Reuses the ImageInstanceSegmentation
        as the return type for convenience.
        """
        category_id_to_category = {c.id: c for c in self._categories}
        for image in self.get_images():
            mask = self._get_mask(image_filepath=image.filename)
            category_ids_in_mask = mask.category_ids()
            objects = [
                SingleInstanceSegmentation(
                    category=category_id_to_category[cid],
                    segmentation=mask.to_binary_mask(category_id=cid),
                )
                for cid in sorted(category_ids_in_mask)
            ]
            yield ImageInstanceSegmentation(
                image=image,
                objects=objects,
            )

    def _get_mask(self, image_filepath: str) -> SemanticSegmentationMask:
        # Validate image exists in our index.
        image_obj = self._filename_to_image.get(image_filepath)
        if image_obj is None:
            raise ValueError(
                f"Unknown image filepath {image_filepath}. Use one returned by get_images()."
            )

        mask_path = self._masks_dir / Path(image_filepath).with_suffix(".png")
        if not mask_path.is_file():
            raise ValueError(
                f"Mask PNG not found for image '{image_filepath}': {mask_path}"
            )

        # Load and validate mask by shape and value set.
        with PILImage.open(mask_path) as mimg:
            mask_np: NDArray[np.int_] = np.asarray(mimg, dtype=np.int_)
        _validate_mask(
            image_obj=image_obj,
            mask_np=mask_np,
            valid_class_ids={c.id for c in self._categories},
        )

        return SemanticSegmentationMask.from_array(array=mask_np)


def _validate_mask(
    image_obj: Image, mask_np: NDArray[np.int_], valid_class_ids: set[int]
) -> None:
    """Validate a semantic segmentation mask against an image and categories.

    Args:
        image_obj: The image metadata with filename, width, and height used for shape validation.
        mask_np: The mask as a 2D numpy array with integer class IDs.
        valid_class_ids: The set of allowed class IDs that may appear in the mask.

    Raises:
        ValueError: If the mask is not 2D, does not match the image size, or contains
            class IDs not present in `valid_class_ids`.
    """
    if mask_np.ndim != 2:
        raise ValueError(
            f"Mask must be 2D (H, W) for: {image_obj.filename}. Got shape {mask_np.shape}"
        )

    mh, mw = int(mask_np.shape[0]), int(mask_np.shape[1])
    if (mw, mh) != (image_obj.width, image_obj.height):
        raise ValueError(
            f"Mask shape must match image dimensions for '{image_obj.filename}': "
            f"mask (W,H)=({mw},{mh}) vs image (W,H)=({image_obj.width},{image_obj.height})"
        )

    uniques = np.unique(mask_np.astype(int))
    unique_values = set(uniques)
    unknown_values = unique_values.difference(valid_class_ids)
    if unknown_values:
        raise ValueError(
            f"Mask contains unknown class ids: {', '.join(map(str, sorted(unknown_values)))}"
        )
