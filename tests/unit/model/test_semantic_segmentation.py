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
