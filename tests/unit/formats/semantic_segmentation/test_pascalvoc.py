from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from labelformat.formats.semantic_segmentation.pascalvoc import (
    PascalVOCSemanticSegmentationInput,
)
from labelformat.model.image import Image
from tests.unit.test_utils import FIXTURES_DIR

FIXTURES_ROOT_PASCALVOC = FIXTURES_DIR / "semantic_segmentation/pascalvoc"
IMAGES_DIR = FIXTURES_ROOT_PASCALVOC / "JPEGImages"
MASKS_DIR = FIXTURES_ROOT_PASCALVOC / "SegmentationClass"
CLASS_MAP_PATH = FIXTURES_ROOT_PASCALVOC / "class_id_to_name.json"


def _load_class_mapping_int_keys() -> Dict[int, str]:
    with CLASS_MAP_PATH.open("r") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


class TestPascalVOCSemanticSegmentationInput:
    def test_from_dirs__builds_categories_and_images(self) -> None:
        mapping = _load_class_mapping_int_keys()
        ds = PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, class_id_to_name=mapping
        )

        # Categories contain the same ids and names as mapping
        cats = list(ds.get_categories())
        loaded_mapping = {c.id: c.name for c in cats}
        assert loaded_mapping == mapping

        # Images are discovered from the folder
        imgs = list(ds.get_images())
        assert len(imgs) == 2
        filenames = {img.filename for img in imgs}
        assert filenames == {"2007_000032.jpg", "2007_000033.jpg"}

    def test_get_mask__returns_int2d_and_matches_image_shape(self) -> None:
        mapping = _load_class_mapping_int_keys()
        ds = PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, class_id_to_name=mapping
        )

        for img in ds.get_images():
            m = ds.get_mask(img.filename)
            assert m.array.ndim == 2
            assert np.issubdtype(m.array.dtype, np.integer)
            assert m.array.shape == (img.height, img.width)
