"""Tests for mask utilities."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from labelformat.mask_utils import (
    binarize_mask,
    extract_instance_masks,
    mask_to_binary_mask_segmentation,
    mask_to_multipolygon,
    match_image_mask_pairs,
)


class TestBinarizeMask:
    def test_binarize_with_threshold(self, tmp_path: Path) -> None:
        """Test binarization with a fixed threshold."""
        # Create a simple grayscale image
        mask: NDArray[np.uint8] = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        mask_path = tmp_path / "test_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # Binarize with threshold 100
        result = binarize_mask(mask_path, threshold=100)
        expected: NDArray[np.uint8] = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_binarize_with_otsu(self, tmp_path: Path) -> None:
        """Test binarization with Otsu automatic threshold."""
        # Create a bimodal image suitable for Otsu
        mask: NDArray[np.uint8] = np.array([[50, 50], [200, 200]], dtype=np.uint8)
        mask_path = tmp_path / "test_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # Binarize with Otsu (None threshold)
        result = binarize_mask(mask_path, threshold=None)
        expected: NDArray[np.uint8] = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_binarize_file_not_found(self) -> None:
        """Test error handling for missing file."""
        with pytest.raises(RuntimeError, match="Failed to read mask image"):
            binarize_mask(Path("nonexistent.png"))


class TestExtractInstanceMasks:
    def test_single_instance(self) -> None:
        """Test extraction of a single connected component."""
        mask: NDArray[np.uint8] = np.array(
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        instances = extract_instance_masks(mask)
        assert len(instances) == 1
        expected: NDArray[np.uint8] = np.array(
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        np.testing.assert_array_equal(instances[0], expected)

    def test_multiple_instances(self) -> None:
        """Test extraction of multiple connected components."""
        mask: NDArray[np.uint8] = np.array(
            [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]], dtype=np.uint8
        )

        instances = extract_instance_masks(mask)
        assert len(instances) == 2

        # First instance (left blob)
        expected1: NDArray[np.uint8] = np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        # Second instance (right blob)
        expected2: NDArray[np.uint8] = np.array(
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8
        )

        # Check that we get both instances (order may vary)
        found_instance1 = any(np.array_equal(inst, expected1) for inst in instances)
        found_instance2 = any(np.array_equal(inst, expected2) for inst in instances)
        assert found_instance1 and found_instance2

    def test_empty_mask(self) -> None:
        """Test extraction from empty mask."""
        mask: NDArray[np.uint8] = np.zeros((3, 3), dtype=np.uint8)
        instances = extract_instance_masks(mask)
        assert len(instances) == 0


class TestMaskToMultipolygon:
    def test_simple_rectangle(self) -> None:
        """Test conversion of rectangular mask to multipolygon."""
        mask: NDArray[np.uint8] = np.array(
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        multipolygon = mask_to_multipolygon(mask)
        assert len(multipolygon.polygons) == 1
        # Should have 4 corner points for rectangle
        assert len(multipolygon.polygons[0]) >= 4


class TestMaskToBinaryMaskSegmentation:
    def test_simple_mask(self) -> None:
        """Test conversion to BinaryMaskSegmentation."""
        mask: NDArray[np.uint8] = np.array(
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8
        )

        binary_seg = mask_to_binary_mask_segmentation(mask)
        assert binary_seg.width == 3
        assert binary_seg.height == 3

        # Verify we can reconstruct the mask
        reconstructed = binary_seg.get_binary_mask()
        np.testing.assert_array_equal(reconstructed, mask.astype(np.int_))


class TestMatchImageMaskPairs:
    def test_stem_matching(self, tmp_path: Path) -> None:
        """Test matching files by stem name."""
        # Create test files
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()

        # Create image files
        (tmp_path / "images" / "img001.jpg").touch()
        (tmp_path / "images" / "img002.jpg").touch()

        # Create matching mask files
        (tmp_path / "masks" / "img001.png").touch()
        (tmp_path / "masks" / "img002.png").touch()

        pairs = match_image_mask_pairs(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="stem",
        )

        assert len(pairs) == 2
        # Verify both pairs are found
        pair_stems = {(img.stem, mask.stem) for img, mask in pairs}
        expected_stems = {("img001", "img001"), ("img002", "img002")}
        assert pair_stems == expected_stems

    def test_index_matching(self, tmp_path: Path) -> None:
        """Test matching files by sorted index."""
        # Create test files
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()

        # Create image files
        (tmp_path / "images" / "a.jpg").touch()
        (tmp_path / "images" / "b.jpg").touch()

        # Create mask files
        (tmp_path / "masks" / "x.png").touch()
        (tmp_path / "masks" / "y.png").touch()

        pairs = match_image_mask_pairs(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="index",
        )

        assert len(pairs) == 2
        # Should pair in sorted order: a.jpg -> x.png, b.jpg -> y.png
        assert pairs[0][0].name == "a.jpg"
        assert pairs[0][1].name == "x.png"
        assert pairs[1][0].name == "b.jpg"
        assert pairs[1][1].name == "y.png"

    def test_regex_matching(self, tmp_path: Path) -> None:
        """Test matching files by regex extracted IDs."""
        # Create test files
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()

        # Create image files with numeric IDs
        (tmp_path / "images" / "crack_001.jpg").touch()
        (tmp_path / "images" / "crack_002.jpg").touch()

        # Create mask files with same numeric IDs
        (tmp_path / "masks" / "mask_001.png").touch()
        (tmp_path / "masks" / "mask_002.png").touch()

        pairs = match_image_mask_pairs(
            image_glob="images/*.jpg",
            mask_glob="masks/*.png",
            base_path=tmp_path,
            pairing_mode="regex",
        )

        assert len(pairs) == 2
        # Verify matching by ID
        for img_path, mask_path in pairs:
            # Extract IDs and verify they match
            img_id = img_path.stem.split("_")[1]  # crack_001 -> 001
            mask_id = mask_path.stem.split("_")[1]  # mask_001 -> 001
            assert img_id == mask_id

    def test_no_matching_files(self, tmp_path: Path) -> None:
        """Test error when no matching files found."""
        with pytest.raises(ValueError, match="No images found"):
            match_image_mask_pairs(
                image_glob="*.jpg",
                mask_glob="*.png",
                base_path=tmp_path,
                pairing_mode="stem",
            )

    def test_invalid_pairing_mode(self, tmp_path: Path) -> None:
        """Test error for invalid pairing mode."""
        # Create dummy files first so we can test pairing mode validation
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.png").touch()

        with pytest.raises(ValueError, match="Invalid pairing mode"):
            match_image_mask_pairs(
                image_glob="*.jpg",
                mask_glob="*.png",
                base_path=tmp_path,
                pairing_mode="invalid",
            )
