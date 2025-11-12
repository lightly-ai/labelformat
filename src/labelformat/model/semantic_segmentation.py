from __future__ import annotations

"""Semantic segmentation core types and input interface.

Design goals:
- Keep the API minimal and numpy-first (no RLE here).
- Avoid CLI wiring for now (TODO: add CLI integration later if needed).
- Mirror patterns from existing model modules while focusing on semseg specifics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from labelformat.model.category import Category
from labelformat.model.image import Image


@dataclass(frozen=True)
class SemSegMask:
    """A semantic segmentation mask.

    Internally represented as a 2D numpy array of integer class IDs with shape (H, W).
    The optional ``ignore_index`` indicates a label value that should be treated as
    "void" and excluded from categories/training, but allowed to appear in the mask.
    """

    array: NDArray[np.int_]
    ignore_index: int | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not isinstance(self.array, np.ndarray):
            raise TypeError("SemSegMask.array must be a numpy ndarray.")
        if self.array.ndim != 2:
            raise ValueError("SemSegMask.array must be 2D with shape (H, W).")
        if not np.issubdtype(self.array.dtype, np.integer):
            raise TypeError("SemSegMask.array must have an integer dtype.")


class SemanticSegmentationInput(ABC):
    """Abstract interface for semantic segmentation datasets.

    Notes:
    - No CLI interface is provided yet. TODO: add CLI integration if needed later.
    - ``get_images`` returns relative filepaths as strings.
    - ``get_mask`` returns a SemSegMask with class-id values per pixel.
    """

    @abstractmethod
    def get_categories(self) -> Iterable[Category]:
        """Returns the list of trainable categories.

        Implementations should typically exclude ``ignore_index`` from categories.
        """

    @abstractmethod
    def get_images(self) -> Iterable[Image]:
        """Yields Image objects for the dataset images (relative filenames)."""

    @abstractmethod
    def get_mask(self, image_filepath: str) -> SemSegMask:
        """Returns the semantic mask for a given image filepath (relative)."""


# Intentionally no Output class here. The consumer can convert and save as needed.
