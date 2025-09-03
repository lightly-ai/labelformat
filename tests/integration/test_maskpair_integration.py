"""Integration tests for maskpair format."""
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from labelformat.formats.coco import COCOInstanceSegmentationOutput
from labelformat.formats.maskpair import MaskPairInstanceSegmentationInput


def create_test_data(base_path: Path) -> None:
    """Create test images and masks for integration testing."""
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # Create simple test images and corresponding masks
    for i in range(3):
        # Create a simple colored image
        image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        image_path = images_dir / f"test_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)

        # Create a corresponding binary mask with some shapes
        mask = np.zeros((100, 100), dtype=np.uint8)

        if i == 0:
            # Single rectangle
            mask[20:40, 20:40] = 255
        elif i == 1:
            # Two separate rectangles
            mask[10:30, 10:30] = 255
            mask[50:70, 50:70] = 255
        else:
            # Circle-like shape
            cv2.circle(mask, (50, 50), 20, 255, -1)

        mask_path = masks_dir / f"test_{i:03d}.png"
        cv2.imwrite(str(mask_path), mask)


class TestMaskPairIntegration:
    def test_maskpair_to_coco_conversion(self, tmp_path: Path) -> None:
        """Test complete maskpair to COCO conversion pipeline."""
        # Create test data
        create_test_data(tmp_path)

        # Create maskpair input
        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
            category_names="defect",
            threshold=128,
            min_area=50.0,
            segmentation_type="polygon",
        )

        # Convert to COCO format
        coco_output_path = tmp_path / "output.json"
        coco_output = COCOInstanceSegmentationOutput(output_file=coco_output_path)
        coco_output.save(label_input=maskpair_input)

        # Verify COCO output
        assert coco_output_path.exists()

        with coco_output_path.open() as f:
            coco_data = json.load(f)

        # Verify structure
        assert "images" in coco_data
        assert "categories" in coco_data
        assert "annotations" in coco_data

        # Should have 3 images
        assert len(coco_data["images"]) == 3

        # Should have 1 category
        assert len(coco_data["categories"]) == 1
        assert coco_data["categories"][0]["name"] == "defect"

        # Should have annotations (at least some instances)
        assert len(coco_data["annotations"]) > 0

        # Verify image properties
        for img in coco_data["images"]:
            assert img["width"] == 100
            assert img["height"] == 100
            assert "test_" in img["file_name"]

        # Verify annotations have required fields
        for ann in coco_data["annotations"]:
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann
            assert "segmentation" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0  # polygon format

    def test_different_pairing_modes(self, tmp_path: Path) -> None:
        """Test different pairing strategies."""
        # Create test data with specific naming
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create files with different naming conventions
        image_files = ["crack_001.jpg", "crack_002.jpg"]
        mask_files = ["mask_001.png", "mask_002.png"]

        for img_name, mask_name in zip(image_files, mask_files):
            # Create simple test files
            img = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / img_name), img)

            mask = np.ones((50, 50), dtype=np.uint8) * 255
            cv2.imwrite(str(masks_dir / mask_name), mask)

        # Test regex pairing mode
        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="regex",
            category_names="test",
            min_area=10.0,
        )

        # Should successfully find 2 pairs
        images = list(maskpair_input.get_images())
        assert len(images) == 2

    def test_rle_segmentation_format(self, tmp_path: Path) -> None:
        """Test RLE segmentation output format."""
        create_test_data(tmp_path)

        # Create maskpair input with RLE format
        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
            category_names="defect",
            segmentation_type="rle",  # Use RLE instead of polygon
            min_area=50.0,
        )

        # Convert to COCO format
        coco_output_path = tmp_path / "output_rle.json"
        coco_output = COCOInstanceSegmentationOutput(output_file=coco_output_path)
        coco_output.save(label_input=maskpair_input)

        # Verify COCO output
        with coco_output_path.open() as f:
            coco_data = json.load(f)

        # Verify that annotations use RLE format
        for ann in coco_data["annotations"]:
            assert ann["iscrowd"] == 1  # RLE format
            assert "counts" in ann["segmentation"]
            assert "size" in ann["segmentation"]

    def test_morphological_operations(self, tmp_path: Path) -> None:
        """Test morphological operations on masks."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create image
        image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "test.jpg"), image)

        # Create a noisy mask that would benefit from morphological operations
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        # Add noise
        mask[35, 35] = 0  # Small hole
        mask[25, 25] = 255  # Small isolated pixel
        cv2.imwrite(str(masks_dir / "test.png"), mask)

        # Test with morphological operations
        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
            category_names="test",
            morph_open=3,  # Remove small noise
            morph_close=3,  # Fill small holes
            min_area=100.0,
        )

        labels = list(maskpair_input.get_labels())
        assert len(labels) == 1
        # Should have filtered out noise and kept main object
        assert len(labels[0].objects) > 0

    def test_minimum_area_filtering(self, tmp_path: Path) -> None:
        """Test filtering instances by minimum area."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Create image
        image = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "test.jpg"), image)

        # Create mask with one large and one small instance
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # Large instance (60x60 = 3600 pixels)
        mask[5:10, 5:10] = 255  # Small instance (5x5 = 25 pixels)
        cv2.imwrite(str(masks_dir / "test.png"), mask)

        # Test with high minimum area threshold
        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
            category_names="test",
            min_area=1000.0,  # Should filter out small instance
        )

        labels = list(maskpair_input.get_labels())
        assert len(labels) == 1
        # Should only keep the large instance
        assert len(labels[0].objects) == 1

    def test_invalid_configuration(self, tmp_path: Path) -> None:
        """Test error handling for invalid configurations."""
        # Test with non-existent glob patterns
        with pytest.raises(ValueError, match="No images found"):
            MaskPairInstanceSegmentationInput(
                image_glob="nonexistent/*.jpg",
                mask_glob="nonexistent/*.png",
                base_path=tmp_path,
                pairing_mode="stem",
                category_names="test",
            )

    def test_empty_category_fallback(self, tmp_path: Path) -> None:
        """Test fallback when no category names provided."""
        create_test_data(tmp_path)

        maskpair_input = MaskPairInstanceSegmentationInput(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
            category_names="",  # Empty category names
            min_area=10.0,
        )

        categories = list(maskpair_input.get_categories())
        # Should create a default category
        assert len(categories) > 0
        # Default category should be created in the get_labels method
