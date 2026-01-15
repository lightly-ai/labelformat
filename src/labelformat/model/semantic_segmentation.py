from __future__ import annotations

from typing import List, Optional, Tuple

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

        return cls(category_id_rle=category_id_rle)


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
