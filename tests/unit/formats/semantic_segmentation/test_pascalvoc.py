from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from labelformat.formats.semantic_segmentation import pascalvoc as pascalvoc_module
from labelformat.formats.semantic_segmentation.pascalvoc import (
    PascalVOCSemanticSegmentationInput,
)
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
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
        assert filenames == {"2007_000032.jpg", "subdir/2007_000033.jpg"}

    def test_get_mask__returns_rle_and_matches_image_length(self) -> None:
        mapping = _load_class_mapping_int_keys()
        ds = PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, class_id_to_name=mapping
        )

        for img in ds.get_images():
            mask = ds._get_mask(img.filename)
            length = sum(run_length for _, run_length in mask.category_id_rle)
            assert length == img.width * img.height

    def test_from_dirs__missing_mask_raises(self, tmp_path: Path) -> None:
        masks_tmp = tmp_path / "SegmentationClass"
        masks_tmp.mkdir(parents=True, exist_ok=True)
        # Copy only one of the two masks
        (masks_tmp / "2007_000032.png").write_bytes(
            (MASKS_DIR / "2007_000032.png").read_bytes()
        )

        with pytest.raises(
            ValueError, match=r"Missing mask PNG for image 'subdir/2007_000033\.jpg'"
        ):
            PascalVOCSemanticSegmentationInput.from_dirs(
                images_dir=IMAGES_DIR,
                masks_dir=masks_tmp,
                class_id_to_name=_load_class_mapping_int_keys(),
            )

    def test_get_mask__unknown_image_raises(self) -> None:
        ds = PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=IMAGES_DIR,
            masks_dir=MASKS_DIR,
            class_id_to_name=_load_class_mapping_int_keys(),
        )
        with pytest.raises(
            ValueError,
            match=r"Unknown image filepath does_not_exist\.jpg",
        ):
            ds._get_mask("does_not_exist.jpg")

    def test_get_labels(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        # Create a simple image and mask
        image0_bgr = np.full((3, 4, 3), (255, 0, 0), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "image0.jpg"), image0_bgr)
        mask0 = np.array([[1, 0, 0, 0], [1, 0, 2, 2], [0, 0, 2, 0]], dtype=np.uint8)
        cv2.imwrite(str(masks_dir / "image0.png"), mask0)

        # Create another image and mask
        image1_bgr = np.full((2, 2, 3), (0, 255, 0), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "image1.jpg"), image1_bgr)
        mask1 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        cv2.imwrite(str(masks_dir / "image1.png"), mask1)

        # Create input instance
        label_input = PascalVOCSemanticSegmentationInput.from_dirs(
            images_dir=images_dir,
            masks_dir=masks_dir,
            class_id_to_name={0: "a", 1: "b", 2: "c", 3: "d"},
        )

        # Call get_labels
        labels = sorted(label_input.get_labels(), key=lambda x: x.image.filename)
        assert len(labels) == 2

        # Verify first image labels
        assert labels[0].image.filename == "image0.jpg"
        objects = labels[0].objects
        assert len(objects) == 3
        assert objects[0].category.id == 0
        assert objects[0].category.name == "a"
        assert isinstance(objects[0].segmentation, BinaryMaskSegmentation)
        assert objects[0].segmentation.get_binary_mask().tolist() == [
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [1, 1, 0, 1],
        ]
        assert objects[1].category.id == 1
        assert objects[1].category.name == "b"
        assert isinstance(objects[1].segmentation, BinaryMaskSegmentation)
        assert objects[1].segmentation.get_binary_mask().tolist() == [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        assert objects[2].category.id == 2
        assert objects[2].category.name == "c"
        assert isinstance(objects[2].segmentation, BinaryMaskSegmentation)
        assert objects[2].segmentation.get_binary_mask().tolist() == [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ]

        # Verify second image labels
        assert labels[1].image.filename == "image1.jpg"
        assert len(labels[1].objects) == 1
        obj = labels[1].objects[0]
        assert obj.category.id == 1
        assert obj.category.name == "b"
        assert isinstance(obj.segmentation, BinaryMaskSegmentation)
        assert obj.segmentation.get_binary_mask().tolist() == [
            [1, 1],
            [1, 1],
        ]


def test__validate_mask__unknown_class_value_raises() -> None:
    # Arrange: simple image and a mask with out-of-vocabulary value
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    mask = np.zeros((3, 4), dtype=np.int_)
    mask[0, 0] = 254
    valid_ids = set(_load_class_mapping_int_keys().keys())

    # Act/Assert
    with pytest.raises(ValueError, match=r"Mask contains unknown class ids: 254"):
        pascalvoc_module._validate_mask(
            image_obj=img, mask_np=mask, valid_class_ids=valid_ids
        )


def test__validate_mask__shape_mismatch_raises() -> None:
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    # Wrong shape (2,4) instead of (3,4)
    mask = np.zeros((2, 4), dtype=np.int_)
    valid_ids = set(_load_class_mapping_int_keys().keys())

    with pytest.raises(ValueError, match=r"Mask shape must match image dimensions"):
        pascalvoc_module._validate_mask(
            image_obj=img, mask_np=mask, valid_class_ids=valid_ids
        )


def test__validate_mask__non_2d_mask_raises() -> None:
    img = Image(id=0, filename="foo.jpg", width=4, height=3)
    # 3D array to simulate multi-channel mask
    mask = np.zeros((3, 4, 3), dtype=np.int_)
    valid_ids = set(_load_class_mapping_int_keys().keys())

    with pytest.raises(ValueError, match=r"Mask must be 2D \(H, W\)"):
        pascalvoc_module._validate_mask(
            image_obj=img, mask_np=mask, valid_class_ids=valid_ids
        )
