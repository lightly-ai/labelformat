import numpy as np

from labelformat.model.binary_mask_segmentation import (
    BinaryMaskSegmentation,
    RLEDecoderEncoder,
)
from labelformat.model.bounding_box import BoundingBox


class TestBinaryMaskSegmentation:

    def test_from_binary_mask(self) -> None:
        # Create a binary mask
        binary_mask = np.array([[0, 1], [1, 0]], dtype=np.int_)
        bounding_box = BoundingBox(0, 0, 2, 2)

        binary_mask_segmentation = BinaryMaskSegmentation.from_binary_mask(
            binary_mask=binary_mask, bounding_box=bounding_box
        )
        assert binary_mask_segmentation.width == 2
        assert binary_mask_segmentation.height == 2
        assert binary_mask_segmentation.bounding_box == bounding_box
        assert np.array_equal(binary_mask_segmentation.get_binary_mask(), binary_mask)


class TestRLEDecoderEncoder:

    def test_encode_row_wise_rle(self) -> None:
        binary_mask = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int_)
        rle = RLEDecoderEncoder.encode_row_wise_rle(binary_mask)
        assert rle == [1, 2, 1, 4]

    def test_decode_row_wise_rle(self) -> None:
        rle = [1, 2, 1, 4]
        height = 2
        width = 4
        binary_mask = RLEDecoderEncoder.decode_row_wise_rle(rle, height, width)
        expected_binary_mask = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
        assert np.array_equal(binary_mask, expected_binary_mask)

    def test_encode_column_wise_rle(self) -> None:
        binary_mask = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int_)
        rle = RLEDecoderEncoder.encode_column_wise_rle(binary_mask)
        assert rle == [1, 5, 1, 1]

    def test_decode_column_wise_rle(self) -> None:
        rle = [1, 5, 1, 1]
        height = 2
        width = 4
        binary_mask = RLEDecoderEncoder.decode_column_wise_rle(rle, height, width)
        expected_binary_mask = np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
        assert np.array_equal(binary_mask, expected_binary_mask)

    def test_inverse__row_wise(self) -> None:
        mask = np.random.randint(0, 2, (42, 9), dtype=np.int_)

        rle = RLEDecoderEncoder.encode_row_wise_rle(mask)
        mask_inverse_row_wise = RLEDecoderEncoder.decode_row_wise_rle(
            rle, mask.shape[0], mask.shape[1]
        )
        assert np.array_equal(mask, mask_inverse_row_wise)

    def test_inverse__column_wise(self) -> None:
        mask = np.random.randint(0, 2, (42, 9), dtype=np.int_)

        rle = RLEDecoderEncoder.encode_column_wise_rle(mask)
        mask_inverse_column_wise = RLEDecoderEncoder.decode_column_wise_rle(
            rle, mask.shape[0], mask.shape[1]
        )
        assert np.array_equal(mask, mask_inverse_column_wise)
