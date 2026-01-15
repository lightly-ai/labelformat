from __future__ import annotations

from typing import List, Optional, Tuple

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.instance_segmentation import SingleInstanceSegmentation

"""Semantic segmentation core types and input interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from labelformat.model.category import Category
from labelformat.model.image import Image


@dataclass
class SemanticSegmentationMask:
    """Semantic segmentation mask with integer class IDs.

    The mask is stored as a 2D numpy array of integer class IDs with shape (H, W).

    Args:
        array: The 2D numpy array with integer class IDs of shape (H, W).
    """

    category_id_rle: List[Tuple[int, int]]
    """The mask as a run-length encoding (RLE) list of (category_id, run_length) tuples."""
    width: int
    height: int

    @classmethod
    def from_array(cls, array: NDArray[np.int_]) -> "SemanticSegmentationMask":
        """Create a SemanticSegmentationMask from a 2D numpy array."""
        if array.ndim != 2:
            raise ValueError("SemSegMask.array must be 2D with shape (H, W).")

        category_id_rle: List[Tuple[int, int]] = []

        cur_cat_id: Optional[int] = None
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


class SemanticSegmentationInput(ABC):

    # TODO(Malte, 11/2025): Add a CLI interface later if needed.

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_images(self) -> Iterable[Image]:
        raise NotImplementedError()

    @abstractmethod
    def get_mask(self, image_filepath: str) -> SemanticSegmentationMask:
        raise NotImplementedError()
