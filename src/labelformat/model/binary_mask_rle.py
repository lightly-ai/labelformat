from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from labelformat.model.bounding_box import BoundingBox


@dataclass(frozen=True)
class BinaryMaskSegmentation:
    """
    A binary mask.
    Internally, the mask is represented as a run-length encoding (RLE) format.
    """

    _rle_row_wise: list[int]
    width: int
    height: int
    bounding_box: BoundingBox

    @classmethod
    def from_binary_mask(
        cls, binary_mask: NDArray[np.int_], bounding_box: BoundingBox
    ) -> "BinaryMaskSegmentation":
        """
        Create a BinaryMaskSegmentation instance from a binary mask (2D numpy array)
        by converting it to RLE format.
        """
        if not isinstance(binary_mask, np.ndarray):
            raise ValueError("Binary mask must be a numpy array.")
        if binary_mask.ndim != 2:
            raise ValueError("Binary mask must be a 2D array.")
        height, width = binary_mask.shape

        rle_row_wise = RLEDecoderEncoder.encode_row_wise_rle(binary_mask)
        return cls(
            _rle_row_wise=rle_row_wise,
            width=width,
            height=height,
            bounding_box=bounding_box,
        )

    def get_binary_mask(self) -> NDArray[np.int_]:
        """
        Get the binary mask (2D numpy array) from the RLE format.
        """
        return RLEDecoderEncoder.decode_row_wise_rle(
            self._rle_row_wise, self.height, self.width
        )


class RLEDecoderEncoder:
    """
    A class for encoding and decoding binary masks using run-length encoding (RLE).
    This class provides methods to encode a binary mask into RLE format and
    decode an RLE list back into a binary mask.

    The encoding and decoding can be done in both row-major and column-major order.
    """

    @staticmethod
    def encode_row_wise_rle(binary_mask: NDArray[np.int_]) -> list[int]:
        # Encodes a binary mask using row-major order.
        flat = np.concatenate(([-1], binary_mask.ravel(order="C"), [-1]))
        borders = np.nonzero(np.diff(flat))[0]
        rle = np.diff(borders)
        if flat[1]:
            rle = np.concatenate(([0], rle))
        rle_list: list[int] = rle.tolist()
        return rle_list

    @staticmethod
    def encode_column_wise_rle(binary_mask: NDArray[np.int_]) -> list[int]:
        # Encodes a binary mask using column-major order.
        flat = np.concatenate(([-1], binary_mask.ravel(order="F"), [-1]))
        borders = np.nonzero(np.diff(flat))[0]
        rle = np.diff(borders)
        if flat[1]:
            rle = np.concatenate(([0], rle))
        rle_list: list[int] = rle.tolist()
        return rle_list

    @staticmethod
    def decode_row_wise_rle(
        rle: list[int], height: int, width: int
    ) -> NDArray[np.int_]:
        # Decodes a row-major run-length encoded list into a 2D binary mask.
        run_val = 0
        decoded = []
        for count in rle:
            decoded.extend([run_val] * count)
            run_val = 1 - run_val
        return np.array(decoded, dtype=np.int_).reshape((height, width), order="C")

    @staticmethod
    def decode_column_wise_rle(
        rle: list[int], height: int, width: int
    ) -> NDArray[np.int_]:
        # Decodes a column-major run-length encoded list into a 2D binary mask.
        run_val = 0
        decoded = []
        for count in rle:
            decoded.extend([run_val] * count)
            run_val = 1 - run_val
        return np.array(decoded, dtype=np.int_).reshape((height, width), order="F")
