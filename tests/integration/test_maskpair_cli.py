"""Test CLI integration for maskpair format."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def create_cli_test_data(base_path: Path) -> None:
    """Create test data for CLI testing."""
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # Create a few test image/mask pairs
    for i in range(2):
        # Create simple test image
        image = np.random.randint(50, 200, (80, 80, 3), dtype=np.uint8)
        image_path = images_dir / f"sample_{i:02d}.jpg"
        cv2.imwrite(str(image_path), image)

        # Create corresponding mask
        mask: NDArray[np.uint8] = np.zeros((80, 80), dtype=np.uint8)
        # Add some rectangular regions
        mask[20:60, 20:40] = 255
        if i == 1:
            mask[20:40, 50:70] = 255  # Add second region for second image

        mask_path = masks_dir / f"sample_{i:02d}.png"
        cv2.imwrite(str(mask_path), mask)


def test_cli_maskpair_to_coco(tmp_path: Path) -> None:
    """Test complete CLI conversion from maskpair to COCO."""
    create_cli_test_data(tmp_path)

    output_file = tmp_path / "output.json"

    # Run CLI command
    cmd = [
        "labelformat",
        "convert",
        "--task",
        "instance-segmentation",
        "--input-format",
        "maskpair",
        "--image-glob",
        "images/*.jpg",
        "--mask-glob",
        "masks/*.png",
        "--base-path",
        str(tmp_path),
        "--category-names",
        "crack,defect",
        "--pairing-mode",
        "stem",
        "--segmentation-type",
        "polygon",
        "--min-area",
        "100",
        "--output-format",
        "coco",
        "--output-file",
        str(output_file),
    ]

    # Execute CLI command
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)

    # Verify command succeeded
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"

    # Verify output file was created
    assert output_file.exists()

    # Verify COCO format structure
    with output_file.open() as f:
        coco_data = json.load(f)

    assert "images" in coco_data
    assert "categories" in coco_data
    assert "annotations" in coco_data

    # Should have 2 images
    assert len(coco_data["images"]) == 2

    # Should have 2 categories (crack, defect)
    assert len(coco_data["categories"]) == 2

    # Should have some annotations
    assert len(coco_data["annotations"]) > 0

    # Verify polygon format
    for ann in coco_data["annotations"]:
        assert ann["iscrowd"] == 0  # polygon format
        assert isinstance(ann["segmentation"], list)


def test_cli_maskpair_to_yolov8(tmp_path: Path) -> None:
    """Test CLI conversion from maskpair to YOLOv8."""
    create_cli_test_data(tmp_path)

    output_file = tmp_path / "data.yaml"

    # Run CLI command
    cmd = [
        "labelformat",
        "convert",
        "--task",
        "instance-segmentation",
        "--input-format",
        "maskpair",
        "--image-glob",
        "images/*.jpg",
        "--mask-glob",
        "masks/*.png",
        "--base-path",
        str(tmp_path),
        "--category-names",
        "object",
        "--pairing-mode",
        "stem",
        "--segmentation-type",
        "polygon",
        "--output-format",
        "yolov8",
        "--output-file",
        str(output_file),
        "--output-split",
        "train",
    ]

    # Execute CLI command
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)

    # Verify command succeeded
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"

    # Verify YOLOv8 structure was created
    assert output_file.exists()
    assert (tmp_path / "labels").exists()


def test_cli_help_shows_maskpair_options() -> None:
    """Test that CLI help shows maskpair-specific options."""
    result = subprocess.run(
        [
            "labelformat",
            "convert",
            "--task",
            "instance-segmentation",
            "--input-format",
            "maskpair",
            "--output-format",
            "coco",
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    # Verify help includes our custom options
    assert "--image-glob" in result.stdout
    assert "--mask-glob" in result.stdout
    assert "--pairing-mode" in result.stdout
    assert "--category-names" in result.stdout
    assert "--segmentation-type" in result.stdout
    assert "--min-area" in result.stdout
    assert "--threshold" in result.stdout
