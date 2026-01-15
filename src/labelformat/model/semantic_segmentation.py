from __future__ import annotations

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation

"""Semantic segmentation core types and input interface.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SemanticSegmentationMask:
    """Semantic segmentation mask with integer class IDs.

    For internal purposes only, interface might change between minor versions!

    The mask is stored as multiclass run-length encoding (RLE).
    """

    category_id_rle: list[tuple[int, int]]
    """The mask as a run-length encoding (RLE) list of (category_id, run_length) tuples."""
    width: int
    """Width of the mask in pixels."""
    height: int
    """Height of the mask in pixels."""

    @classmethod
    def from_array(cls, array: NDArray[np.int_]) -> "SemanticSegmentationMask":
        """Create a SemanticSegmentationMask from a 2D numpy array."""
        if array.ndim != 2:
            raise ValueError("SemSegMask.array must be 2D with shape (H, W).")

        category_id_rle: list[tuple[int, int]] = []

        cur_cat_id: int | None = None
        cur_run_length = 0
        for cat_id in array.flatten():
            if cat_id == cur_cat_id:
                cur_run_length += 1
            else:
                if cur_cat_id is not None:
                    category_id_rle.append((cur_cat_id, cur_run_length))
                cur_cat_id = cat_id
                cur_run_length = 1
        if cur_cat_id is not None:
            category_id_rle.append((cur_cat_id, cur_run_length))

        return cls(
            category_id_rle=category_id_rle, width=array.shape[1], height=array.shape[0]
        )

    def to_binary_mask(self, category_id: int) -> BinaryMaskSegmentation:
        """Get a binary mask for a given category ID."""
        binary_rle = []

        symbol = 0
        run_length = 0
        for cat_id, cur_run_length in self.category_id_rle:
            cur_symbol = 1 if cat_id == category_id else 0
            if symbol == cur_symbol:
                run_length += cur_run_length
            else:
                binary_rle.append(run_length)
                symbol = cur_symbol
                run_length = cur_run_length

        binary_rle.append(run_length)
        return BinaryMaskSegmentation.from_rle(
            rle_row_wise=binary_rle,
            width=self.width,
            height=self.height,
        )

    def category_ids(self) -> set[int]:
        """Get the set of category IDs present in the mask."""
        return {cat_id for cat_id, _ in self.category_id_rle}
