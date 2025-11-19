from __future__ import annotations

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

    array: NDArray[np.int_]

    def __post_init__(self) -> None:
        if self.array.ndim != 2:
            raise ValueError("SemSegMask.array must be 2D with shape (H, W).")


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
