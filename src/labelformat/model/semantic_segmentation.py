from __future__ import annotations

"""Semantic segmentation core types and input interface.

Design goals:
- Keep the API minimal and numpy-first (no RLE here).
- Avoid CLI wiring for now (TODO: add CLI integration later if needed).
- Mirror patterns from existing model modules while focusing on semseg specifics.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from labelformat.model.category import Category
from labelformat.model.image import Image


@dataclass(frozen=True)
class SemSegMask:
    """Semantic segmentation mask with integer class IDs.

    The mask is stored as a 2D numpy array of integer class IDs with shape (H, W).

    Args:
        array: The 2D numpy array with integer class IDs of shape (H, W).
    """

    array: NDArray[np.int_]

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray):
            raise TypeError("SemSegMask.array must be a numpy ndarray.")
        if self.array.ndim != 2:
            raise ValueError("SemSegMask.array must be 2D with shape (H, W).")
        if not np.issubdtype(self.array.dtype, np.integer):
            raise TypeError("SemSegMask.array must have an integer dtype.")


class SemanticSegmentationInput(ABC):

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        raise NotImplementedError()

    @abstractmethod
    def get_images(self) -> Iterable[Image]:
        raise NotImplementedError()

    @abstractmethod
    def get_mask(self, image_filepath: str) -> SemSegMask:
        raise NotImplementedError()


# Intentionally no Output class here. The consumer can convert and save as needed.
