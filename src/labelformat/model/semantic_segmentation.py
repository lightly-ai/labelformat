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

        if array.size == 0:
            return cls(
                category_id_rle=[],
                width=array.shape[1],
                height=array.shape[0],
            )

        flat = array.ravel()
        change_indices = np.nonzero(flat[:-1] != flat[1:])[0]
        run_starts = np.concatenate(([0], change_indices + 1))
        run_lengths = np.diff(np.concatenate(([0], change_indices + 1, [flat.size])))

        category_ids = flat[run_starts]
        category_id_rle = list(zip(category_ids.tolist(), run_lengths.tolist()))

        return cls(
            category_id_rle=category_id_rle,
            width=array.shape[1],
            height=array.shape[0],
        )

    def to_binary_mask(self, category_id: int) -> BinaryMaskSegmentation:
        """Get a binary mask for a given category ID."""
        if not self.category_id_rle:
            return BinaryMaskSegmentation.from_rle(
                rle_row_wise=[],
                width=self.width,
                height=self.height,
            )

        rle_array = np.asarray(self.category_id_rle, dtype=np.int_)
        cat_ids = rle_array[:, 0]
        run_lengths = rle_array[:, 1]
        symbols = cat_ids == category_id
        change_indices = np.nonzero(symbols[:-1] != symbols[1:])[0] + 1
        run_starts = np.concatenate(([0], change_indices))
        binary_run_lengths = np.add.reduceat(run_lengths, run_starts)
        if symbols[0]:
            binary_run_lengths = np.concatenate(([0], binary_run_lengths))

        return BinaryMaskSegmentation.from_rle(
            rle_row_wise=binary_run_lengths.tolist(),
            width=self.width,
            height=self.height,
        )

    def category_ids(self) -> set[int]:
        """Get the set of category IDs present in the mask."""
        return {cat_id for cat_id, _ in self.category_id_rle}
