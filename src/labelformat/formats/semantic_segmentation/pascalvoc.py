from __future__ import annotations

"""Pascal VOC semantic segmentation input.

Assumptions for this initial version:
- Masks live under a separate directory mirroring the images directory structure.
- For each image at ``images_dir/<rel>.ext``, the mask is at ``masks_dir/<rel>.png``.
- Masks are single-channel PNGs (mode 'L' or 'P') with pixel values equal to class IDs.

TODOs for future extensions (intentionally not implemented yet):
- Support non-PNG masks and custom pairing strategies (e.g., ``*_mask.png``).
- Support palettized/RGB masks with arbitrary pixel_value -> class_id mappings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image as PILImage

from labelformat.model.category import Category
from labelformat.model.semantic_segmentation import (
    SemanticSegmentationInput,
    SemSegMask,
)
from labelformat.utils import IMAGE_EXTENSIONS, get_image_dimensions


@dataclass
class PascalVOCSemanticSegmentationInput(SemanticSegmentationInput):
    _images_dir: Path
    _masks_dir: Path
    _image_relpaths: list[str]
    _categories: list[Category]
    _class_id_to_name: dict[int, str]
    _ignore_index: int | None

    @classmethod
    def from_dirs(
        cls,
        images_dir: Path,
        masks_dir: Path,
        class_id_to_name: Mapping[int | str, str],
        ignore_index: int | None = 255,
    ) -> "PascalVOCSemanticSegmentationInput":
        """Create a PascalVOCSemanticSegmentationInput from directory pairs.

        Args:
            images_dir: Root directory containing images (nested structure allowed).
            masks_dir: Root directory containing PNG masks mirroring images structure.
            class_id_to_name: Mapping of class_id -> class name. Keys may be str or int.
            ignore_index: Optional value used in masks to indicate 'void' (default 255).

        Raises:
            ValueError: If directories are invalid, a mask is missing or not PNG,
                        or if class_id keys cannot be parsed as integers.
        """
        if not images_dir.is_dir():
            raise ValueError(f"Images directory is not a directory: {images_dir}")
        if not masks_dir.is_dir():
            raise ValueError(f"Masks directory is not a directory: {masks_dir}")

        # Normalize class id mapping (coerce keys to int)
        norm_mapping: dict[int, str] = {}
        for k, v in class_id_to_name.items():
            norm_mapping[int(k)] = v

        # Build categories excluding ignore_index
        categories = [
            Category(id=cid, name=norm_mapping[cid])
            for cid in sorted(norm_mapping.keys())
            if ignore_index is None or cid != ignore_index
        ]

        # Collect images and ensure a PNG mask exists for each
        image_relpaths: list[str] = []
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

            image_relpaths.append(rel_str)

        return cls(
            images_dir,
            masks_dir,
            image_relpaths,
            categories,
            norm_mapping,
            ignore_index,
        )

    # --- Public API ---
    def get_categories(self) -> Iterable[Category]:
        # Exclude ignore_index by construction
        return list(self._categories)

    def get_images(self) -> list[str]:
        return list(self._image_relpaths)

    def get_mask(self, image_filepath: str) -> SemSegMask:
        # Validate image exists in our index
        if image_filepath not in self._image_relpaths:
            raise ValueError(
                f"Unknown image filepath (relative): {image_filepath}. "
                "Use one returned by get_images()."
            )

        image_path = self._images_dir / image_filepath
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
        img_w, img_h = get_image_dimensions(image_path)
        mh, mw = int(mask_np.shape[0]), int(mask_np.shape[1])
        if (mw, mh) != (img_w, img_h):
            raise ValueError(
                "Mask shape must match image dimensions for '"
                + image_filepath
                + f"': mask (W,H)=({mw},{mh}) vs image (W,H)=({img_w},{img_h})"
            )

        # 4) Validate values are within known class ids (or ignore_index)
        uniques = np.unique(mask_np)
        # Convert to Python ints for set operations
        unique_values = {int(x) for x in uniques.tolist()}
        valid_class_ids = set(self._class_id_to_name.keys())
        if self._ignore_index is not None:
            valid_or_ignore = valid_class_ids | {self._ignore_index}
        else:
            valid_or_ignore = valid_class_ids
        unknown_values = unique_values.difference(valid_or_ignore)
        if unknown_values:
            # Per requirement: raise an error for absent/extra class IDs
            raise ValueError(
                "Mask contains unknown class ids: "
                + ", ".join(map(str, sorted(unknown_values)))
            )

        return SemSegMask(
            array=mask_np.astype(np.int_), ignore_index=self._ignore_index
        )
