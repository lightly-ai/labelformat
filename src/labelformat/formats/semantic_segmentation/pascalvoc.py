from __future__ import annotations

"""Pascal VOC semantic segmentation input.

Assumptions for this initial version:
- Masks live under a separate directory mirroring the images directory structure.
- For each image at ``images_dir/<rel>.ext``, the mask is at ``masks_dir/<rel>.png``.
- Masks are single-channel PNGs (mode 'L' or 'P') with pixel values equal to class IDs.

TODOs for future extensions (intentionally not implemented yet):
- Support non-PNG masks and custom pairing strategies (e.g., ``*_mask.png``).
- Support palettized/RGB masks with arbitrary pixel_value -> class_id mappings.
- Support an optional ignore_index to allow a void label and optionally exclude it
  from categories.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image as PILImage

from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.semantic_segmentation import (
    SemanticSegmentationInput,
    SemSegMask,
)
from labelformat.utils import IMAGE_EXTENSIONS, get_image_dimensions


@dataclass
class PascalVOCSemanticSegmentationInput(SemanticSegmentationInput):
    _images_dir: Path
    _masks_dir: Path
    _images: list[Image]
    _categories: list[Category]
    # TODO: Add optional ignore_index handling in the future.

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

        # Collect images and ensure a PNG mask exists for each
        images: list[Image] = []
        image_id = 0
        for img_path in sorted(images_dir.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = img_path.relative_to(images_dir)
            rel_str = str(rel)

            # Mask must be a PNG alongside the mirrored relative path
            mask_rel = rel.with_suffix(".png")
            mask_path = masks_dir / mask_rel
            if not mask_path.is_file():
                raise ValueError(
                    "Missing mask PNG for image '"
                    + rel_str
                    + "' at path: "
                    + str(mask_path)
                )

            # Read image dimensions once and store
            width, height = get_image_dimensions(img_path)
            images.append(
                Image(id=image_id, filename=rel_str, width=width, height=height)
            )
            image_id += 1

        return cls(
            images_dir,
            masks_dir,
            images,
            categories,
        )

    def get_categories(self) -> Iterable[Category]:
        return list(self._categories)

    def get_images(self) -> Iterable[Image]:
        yield from self._images

    def get_mask(self, image_filepath: str) -> SemSegMask:
        # Validate image exists in our index
        image_obj = next(
            (img for img in self._images if img.filename == image_filepath), None
        )
        if image_obj is None:
            raise ValueError(
                f"Unknown image filepath (relative): {image_filepath}. "
                "Use one returned by get_images()."
            )

        mask_path = self._masks_dir / Path(image_filepath).with_suffix(".png")

        # 1) Enforce PNG mask
        if mask_path.suffix.lower() != ".png":
            raise ValueError(
                f"Mask must be a PNG file for image '{image_filepath}', got: {mask_path.name}"
            )
        if not mask_path.is_file():
            raise ValueError(
                f"Mask PNG not found for image '{image_filepath}': {mask_path}"
            )

        # 2) Enforce single-channel (accept 'L' or 'P')
        with PILImage.open(mask_path) as mimg:
            mode = mimg.mode
            if mode not in {"L", "P"}:
                raise ValueError(
                    "Mask must be single-channel ('L' or 'P'), "
                    f"but got mode '{mode}' for: {mask_path}. "
                    "TODO: support additional modes."
                )
            mask_np = np.asarray(mimg)

        if mask_np.ndim != 2:
            raise ValueError(
                f"Mask must be 2D (H, W) for: {mask_path}. Got shape {mask_np.shape}"
            )

        # 3) Validate shape matches image dimensions
        img_w, img_h = image_obj.width, image_obj.height
        mh, mw = int(mask_np.shape[0]), int(mask_np.shape[1])
        if (mw, mh) != (img_w, img_h):
            raise ValueError(
                "Mask shape must match image dimensions for '"
                + image_filepath
                + f"': mask (W,H)=({mw},{mh}) vs image (W,H)=({img_w},{img_h})"
            )

        # 4) Validate values are within known class ids
        uniques = np.unique(mask_np)
        # Convert to Python ints for set operations
        unique_values = {int(x) for x in uniques.tolist()}
        valid_class_ids = {cat.id for cat in self._categories}
        # TODO: support optional ignore_index in value validation.
        unknown_values = unique_values.difference(valid_class_ids)
        if unknown_values:
            # Per requirement: raise an error for absent/extra class IDs
            raise ValueError(
                "Mask contains unknown class ids: "
                + ", ".join(map(str, sorted(unknown_values)))
            )

        # TODO: Add ignore_index support to SemSegMask usage if desired.
        return SemSegMask(array=mask_np.astype(np.int_))
