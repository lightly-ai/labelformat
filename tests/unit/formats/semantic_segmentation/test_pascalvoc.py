from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from PIL import Image as PILImage

from labelformat.formats.semantic_segmentation.pascalvoc import (
    PascalVOCSemanticSegmentationInput,
    _validate_mask,
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


def test_from_dirs__builds_categories_and_images() -> None:
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


def test_get_mask__returns_int2d_and_matches_image_shape() -> None:
    mapping = _load_class_mapping_int_keys()
    ds = PascalVOCSemanticSegmentationInput.from_dirs(
        images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, class_id_to_name=mapping
    )

    for img in ds.get_images():
        m = ds.get_mask(img.filename)
        assert m.array.ndim == 2
        assert np.issubdtype(m.array.dtype, np.integer)
        assert m.array.shape == (img.height, img.width)


def test_from_dirs__missing_mask_raises(tmp_path: Path) -> None:
    masks_tmp = tmp_path / "SegmentationClass"
    masks_tmp.mkdir(parents=True, exist_ok=True)
    # Copy only one of the two masks
    (masks_tmp / "2007_000032.png").write_bytes(
        (MASKS_DIR / "2007_000032.png").read_bytes()
    )

    with pytest.raises(
        ValueError, match=r"Missing mask PNG for image '2007_000033\.jpg'"
    ):
        PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=IMAGES_DIR,
            masks_dir=masks_tmp,
            class_id_to_name=_load_class_mapping_int_keys(),
        )


def test_get_mask__unknown_image_raises() -> None:
    ds = PascalVOCSemanticSegmentationInput.from_dirs(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        class_id_to_name=_load_class_mapping_int_keys(),
    )
    with pytest.raises(
        ValueError,
        match=r"Unknown image filepath \(relative\): does_not_exist\.jpg",
    ):
        ds.get_mask("does_not_exist.jpg")


def test_validate_mask__unknown_class_value_raises() -> None:
    # Arrange: simple image and a mask with out-of-vocabulary value
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    mask = np.zeros((3, 4), dtype=np.int_)
    mask[0, 0] = 254
    valid_ids = set(_load_class_mapping_int_keys().keys())

    # Act/Assert
    with pytest.raises(ValueError, match=r"Mask contains unknown class ids: 254"):
        _validate_mask(image_obj=img, mask_np=mask, valid_class_ids=valid_ids)


def test_validate_mask__shape_mismatch_raises() -> None:
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    # Wrong shape (2,4) instead of (3,4)
    mask = np.zeros((2, 4), dtype=np.int_)
    valid_ids = set(_load_class_mapping_int_keys().keys())

    with pytest.raises(ValueError, match=r"Mask shape must match image dimensions"):
        _validate_mask(image_obj=img, mask_np=mask, valid_class_ids=valid_ids)


def test_validate_mask__non_2d_mask_raises() -> None:
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    # 3D array to simulate multi-channel mask
    mask = np.zeros((3, 4, 3), dtype=np.int_)
    valid_ids = set(_load_class_mapping_int_keys().keys())

    with pytest.raises(ValueError, match=r"Mask must be 2D \(H, W\)"):
        _validate_mask(image_obj=img, mask_np=mask, valid_class_ids=valid_ids)
