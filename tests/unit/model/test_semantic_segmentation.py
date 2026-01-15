from __future__ import annotations

import numpy as np

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
        assert binary_mask.get_binary_mask().tolist() == [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
        ]

        binary_mask = mask.to_binary_mask(category_id=2)
        assert binary_mask.get_binary_mask().tolist() == [
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        binary_mask = mask.to_binary_mask(category_id=4)
        assert binary_mask.get_binary_mask().tolist() == [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
