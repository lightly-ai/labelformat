import numpy as np
from numpy.typing import NDArray

from labelformat.model import binary_mask_segmentation
from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox


class TestBinaryMaskSegmentation:
    def test_from_binary_mask(self) -> None:
        # Create a binary mask
        binary_mask: NDArray[np.int_] = np.array([[0, 1], [1, 0]], dtype=np.int_)
        bounding_box = BoundingBox(0, 0, 2, 2)

        binary_mask_segmentation = BinaryMaskSegmentation.from_binary_mask(
            binary_mask=binary_mask, bounding_box=bounding_box
        )
        assert binary_mask_segmentation.width == 2
        assert binary_mask_segmentation.height == 2
        assert binary_mask_segmentation.bounding_box == bounding_box
        assert np.array_equal(binary_mask_segmentation.get_binary_mask(), binary_mask)

    def test_from_rle(self) -> None:
        binary_mask_segmentation = BinaryMaskSegmentation.from_rle(
            rle_row_wise=[1, 1, 4, 2, 1, 3, 2, 1, 5],
            width=5,
            height=4,
            bounding_box=None,
        )
        assert binary_mask_segmentation.width == 5
        assert binary_mask_segmentation.height == 4
        assert binary_mask_segmentation.bounding_box == BoundingBox(0, 0, 5, 3)
        expected: NDArray[np.int_] = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1],
                [1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int_,
        )
        assert np.array_equal(binary_mask_segmentation.get_binary_mask(), expected)

        # Test with provided bounding box
        # The box is larger than the actual mask, but should be preserved
        binary_mask_segmentation = BinaryMaskSegmentation.from_rle(
            rle_row_wise=[6, 3],
            width=3,
            height=3,
            bounding_box=BoundingBox(0, 0, 2, 2),
        )
        assert binary_mask_segmentation.width == 3
        assert binary_mask_segmentation.height == 3
        assert binary_mask_segmentation.bounding_box == BoundingBox(0, 0, 2, 2)
        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.int_,
        )
        assert np.array_equal(binary_mask_segmentation.get_binary_mask(), expected)

    def test_get_rle(self) -> None:
        binary_mask_segmentation = BinaryMaskSegmentation.from_rle(
            rle_row_wise=[1, 2, 3],
            width=3,
            height=2,
            bounding_box=None,
        )
        assert binary_mask_segmentation.get_rle() == [1, 2, 3]


class TestRLEDecoderEncoder:
    def test_encode_row_wise_rle(self) -> None:
        binary_mask: NDArray[np.int_] = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int_
        )
        rle = RLEDecoderEncoder.encode_row_wise_rle(binary_mask)
        assert rle == [1, 2, 1, 4]

    def test_decode_row_wise_rle(self) -> None:
        rle = [1, 2, 1, 4]
        height = 2
        width = 4
        binary_mask = RLEDecoderEncoder.decode_row_wise_rle(rle, height, width)
        expected_binary_mask: NDArray[np.uint8] = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8
        )
        assert np.array_equal(binary_mask, expected_binary_mask)

    def test_encode_column_wise_rle(self) -> None:
        binary_mask: NDArray[np.int_] = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int_
        )
        rle = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
        assert rle == [1, 5, 1, 1]

    def test_decode_column_wise_rle(self) -> None:
        rle = [1, 5, 1, 1]
        height = 2
        width = 4
        binary_mask = RLEDecoderEncoder.decode_column_wise_rle(rle, height, width)
        expected_binary_mask: NDArray[np.uint8] = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8
        )
        assert np.array_equal(binary_mask, expected_binary_mask)

    def test_inverse__row_wise(self) -> None:
        mask: NDArray[np.int_] = np.random.randint(
            0, 2, (42, 9), dtype=np.int32
        ).astype(np.int_)

        rle = RLEDecoderEncoder.encode_row_wise_rle(mask)
        mask_inverse_row_wise = RLEDecoderEncoder.decode_row_wise_rle(
            rle, mask.shape[0], mask.shape[1]
        )
        assert np.array_equal(mask, mask_inverse_row_wise)

    def test_inverse__column_wise(self) -> None:
        mask: NDArray[np.int_] = np.random.randint(
            0, 2, (42, 9), dtype=np.int32
        ).astype(np.int_)

        rle = RLEDecoderEncoder.encode_column_wise_rle(mask)
        mask_inverse_column_wise = RLEDecoderEncoder.decode_column_wise_rle(
            rle, mask.shape[0], mask.shape[1]
        )
        assert np.array_equal(mask, mask_inverse_column_wise)


def test_compute_bbox_from_rle() -> None:
    # 0011
    # 1111
    # 1100
    bbox = binary_mask_segmentation._compute_bbox_from_rle(
        rle_row_wise=[2, 8, 2],
        width=4,
        height=3,
    )
    assert bbox == BoundingBox(xmin=0, ymin=0, xmax=4, ymax=3)

    # 0011
    # 0000
    bbox = binary_mask_segmentation._compute_bbox_from_rle(
        rle_row_wise=[2, 2, 4],
        width=4,
        height=2,
    )
    assert bbox == BoundingBox(xmin=2, ymin=0, xmax=4, ymax=1)

    # 0011
    # 1000
    bbox = binary_mask_segmentation._compute_bbox_from_rle(
        rle_row_wise=[2, 3, 3],
        width=4,
        height=2,
    )
    assert bbox == BoundingBox(xmin=0, ymin=0, xmax=4, ymax=2)

    # 1111
    bbox = binary_mask_segmentation._compute_bbox_from_rle(
        rle_row_wise=[0, 4],
        width=4,
        height=1,
    )
    assert bbox == BoundingBox(xmin=0, ymin=0, xmax=4, ymax=1)
