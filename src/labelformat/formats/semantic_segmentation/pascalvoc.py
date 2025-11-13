from __future__ import annotations

"""Pascal VOC semantic segmentation input.

Assumptions:
- Masks live under a separate directory mirroring the images directory structure.
- For each image at ``images_dir/<rel>.ext``, the mask is at ``masks_dir/<rel>.png``.
- Masks are PNGs with pixel values equal to class IDs.

TODO (Malte, 11/2025)
Support what is already supported in LightlyTrain:
https://docs.lightly.ai/train/stable/semantic_segmentation.html#data
- Support using a template against the image filepath. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-a-template-against-the-image-filepath
- Support using multi-channel masks. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-multi-channel-masks
- Support optional ignore_classes: list[int] that should be ignored during training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.semantic_segmentation import (
    SemanticSegmentationInput,
    SemSegMask,
)
from labelformat.utils import get_images_from_folder


@dataclass
class PascalVOCSemanticSegmentationInput(SemanticSegmentationInput):
    _images_dir: Path
    _masks_dir: Path
    _filename_to_image: dict[str, Image]
    _categories: list[Category]

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

        # Build categories from mapping (no ignore_index handling here)
        categories = [
            Category(id=cid, name=cname) for cid, cname in class_id_to_name.items()
        ]

        # Collect images using helper and ensure a PNG mask exists for each
        images_by_filename: dict[str, Image] = {}
        for img in get_images_from_folder(images_dir):
            mask_path = masks_dir / Path(img.filename).with_suffix(".png")
            if not mask_path.is_file():
                raise ValueError(
                    f"Missing mask PNG for image '{img.filename}' at path: {mask_path}"
                )
            images_by_filename[img.filename] = img

        return cls(images_dir, masks_dir, images_by_filename, categories)

    def get_categories(self) -> Iterable[Category]:
        return list(self._categories)

    def get_images(self) -> Iterable[Image]:
        yield from self._filename_to_image.values()

    def get_mask(self, image_filepath: str) -> SemSegMask:
        # Validate image exists in our index.
        image_obj = self._filename_to_image.get(image_filepath)
        if image_obj is None:
            raise ValueError(
                f"Unknown image filepath (relative): {image_filepath}. Use one returned by get_images()."
            )

        mask_path = self._masks_dir / Path(image_filepath).with_suffix(".png")

        # Enforce PNG mask.
        if mask_path.suffix.lower() != ".png":
            raise ValueError(
                f"Mask must be a PNG file for image '{image_filepath}', got: {mask_path.name}"
            )
        if not mask_path.is_file():
            raise ValueError(
                f"Mask PNG not found for image '{image_filepath}': {mask_path}"
            )

        # Load and validate mask by shape and value set.
        with PILImage.open(mask_path) as mimg:
            mask_np: NDArray[np.int_] = np.asarray(mimg, dtype=np.int_)
        self._validate_mask(image_obj=image_obj, mask_np=mask_np)

        return SemSegMask(array=mask_np)

    def _validate_mask(self, image_obj: Image, mask_np: NDArray[np.int_]) -> None:
        """Validate mask shape and value set; return int-casted mask.

        - Ensures mask is 2D (single-channel).
        - Ensures mask shape matches image dimensions.
        - Ensures mask values are subset of known category IDs.
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

        uniques = np.unique(mask_np)
        unique_values = {int(x) for x in uniques.tolist()}
        valid_class_ids = {cat.id for cat in self._categories}
        unknown_values = unique_values.difference(valid_class_ids)
        if unknown_values:
            raise ValueError(
                f"Mask contains unknown class ids: {', '.join(map(str, sorted(unknown_values)))}"
            )
