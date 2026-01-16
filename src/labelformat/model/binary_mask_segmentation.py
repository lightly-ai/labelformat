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

    @classmethod
    def from_rle(
        cls,
        rle_row_wise: list[int],
        width: int,
        height: int,
        bounding_box: BoundingBox | None = None,
    ) -> "BinaryMaskSegmentation":
        """
        Create a BinaryMaskSegmentation instance from row-wise RLE format.
        """
        if bounding_box is None:
            bounding_box = _compute_bbox_from_rle(
                rle_row_wise=rle_row_wise, width=width, height=height
            )
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

    def get_rle(self) -> list[int]:
        """
        Get the run-length encoding (RLE) of the binary mask in row-wise format.

        The first element corresponds to the number of 0s at the start of the mask.
        If the mask starts with a 1, the first element will be 0. No other zeros can appear.
        """
        return self._rle_row_wise


class RLEDecoderEncoder:
    """
    A class for encoding and decoding binary masks using run-length encoding (RLE).
    This class provides methods to encode a binary mask into RLE format and
    decode an RLE list back into a binary mask.

    The encoding and decoding can be done both row-wise and column-wise.

    Example:
        Consider a binary mask of shape 2x4:
            [[0, 1, 1, 0],
             [1, 1, 1, 1]]
        Row-wise RLE: [1, 2, 1, 4]
        Column-wise RLE: [1, 5, 1, 1]
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


def _compute_bbox_from_rle(
    rle_row_wise: list[int], width: int, height: int
) -> BoundingBox:
    """Compute bounding box from row-wise RLE.

    Scans through the RLE and tracks the min/max x/y coordinates of the '1' pixels.
    The time complexity is O(len(rle_row_wise)).
    """
    xmin = width
    ymin = height
    xmax = 0
    ymax = 0

    x = 0
    y = 0
    next_symbol = 0
    for run_length in rle_row_wise:
        if next_symbol == 1:
            # Compute coordinates for the end of the run
            run_end_x = x + run_length - 1
            run_end_y = y
            if run_end_x >= width:
                run_end_y += run_end_x // width
                run_end_x = run_end_x % width

            # Update bounding box
            ymin = min(ymin, y)
            ymax = max(ymax, run_end_y)
            if run_end_y > y:
                xmin = 0
                xmax = width - 1
            else:
                xmin = min(xmin, x)
                xmax = max(xmax, run_end_x)

        # Compute coordinates for the start of the next run
        x += run_length
        if x >= width:
            y += x // width
            x = x % width

        next_symbol = 1 - next_symbol

    return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
