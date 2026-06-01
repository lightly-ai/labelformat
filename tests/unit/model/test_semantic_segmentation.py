from __future__ import annotations

import numpy as np

from labelformat.model.bounding_box import BoundingBox
from labelformat.model.semantic_segmentation import SemanticSegmentationMask


class TestSemanticSegmentationMask:
    def test_from_array(self) -> None:
        array = np.array(
            [
                [1, 1, 2, 2],
                [2, 1, 1, 1],
                [3, 3, 3, 3],
            ],
            dtype=np.int_,
        )
        expected_rle = [
            (1, 2),
            (2, 3),
            (1, 3),
            (3, 4),
        ]
        mask = SemanticSegmentationMask.from_array(array=array)
        assert mask.category_id_rle == expected_rle
        assert mask.width == 4
        assert mask.height == 3

    def test_to_binary_mask(self) -> None:
        mask = SemanticSegmentationMask.from_array(
            array=np.array(
                [
                    [1, 1, 2, 2],
                    [2, 1, 1, 1],
                    [3, 3, 3, 3],
                ],
                dtype=np.int_,
            )
        )
        binary_mask = mask.to_binary_mask(category_id=1)
        assert binary_mask.get_rle() == [0, 2, 3, 3, 4]
        assert binary_mask.bounding_box == BoundingBox(0, 0, 4, 2)
        assert binary_mask.get_binary_mask().tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
        ]

        binary_mask = mask.to_binary_mask(category_id=2)
        assert binary_mask.get_rle() == [2, 3, 7]
        assert binary_mask.bounding_box == BoundingBox(0, 0, 4, 2)
        assert binary_mask.get_binary_mask().tolist() == [
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        binary_mask = mask.to_binary_mask(category_id=4)
        assert binary_mask.get_rle() == [12]
        assert binary_mask.bounding_box == BoundingBox(4, 3, 0, 0)
        assert binary_mask.get_binary_mask().tolist() == [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

    def test__empty_mask(self) -> None:
        mask = SemanticSegmentationMask.from_array(
            array=np.empty((0, 3), dtype=np.int_)
        )
        assert mask.category_id_rle == []
        assert mask.width == 3
        assert mask.height == 0
        assert mask.category_ids() == set()

        binary_mask = mask.to_binary_mask(category_id=1)
        assert binary_mask.get_rle() == []
        assert binary_mask.bounding_box == BoundingBox(3, 0, 0, 0)
        assert binary_mask.get_binary_mask().shape == (0, 3)

    def test_category_ids(self) -> None:
        mask = SemanticSegmentationMask.from_array(
            array=np.array(
                [
                    [1, 1, 4],
                    [4, 1, 1],
                ],
                dtype=np.int_,
            )
        )
        assert mask.category_ids() == {1, 4}
