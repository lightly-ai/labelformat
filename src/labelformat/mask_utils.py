"""Utilities for processing binary masks for segmentation tasks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.multipolygon import MultiPolygon, Point


def binarize_mask(
    mask_path: Path,
    threshold: int | None = None,
    morph_open: int = 0,
    morph_close: int = 0,
) -> NDArray[np.uint8]:
    """Read a mask image, threshold to binary, apply optional morphology.
    
    Args:
        mask_path: Path to the mask image file
        threshold: Binarization threshold 0-255, None for Otsu automatic
        morph_open: Size of morphological opening kernel in pixels
        morph_close: Size of morphological closing kernel in pixels
        
    Returns:
        Binary mask with values 0 or 1 (uint8)
        
    Raises:
        RuntimeError: If mask image cannot be read
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read mask image: {mask_path}")

    if threshold is None:
        # Use Otsu's automatic threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Optional morphological operations to clean up artifacts
    if morph_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if morph_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Convert to binary 0/1 values
    binary_mask: NDArray[np.uint8] = np.where(binary > 0, 1, 0).astype(np.uint8)
    
    # If foreground covers most of the image, invert (assuming background should be dominant)
    if binary_mask.mean() > 0.5:
        binary_mask = (1 - binary_mask).astype(np.uint8)
        
    return binary_mask


def extract_instance_masks(binary_mask: NDArray[np.uint8]) -> List[NDArray[np.uint8]]:
    """Split a binary mask into connected components (instances).
    
    Args:
        binary_mask: Binary mask with 0/1 values
        
    Returns:
        List of binary masks, one per connected component instance
    """
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    instance_masks = []
    
    for label_id in range(1, num_labels):  # Skip background (label 0)
        instance_mask: NDArray[np.uint8] = np.where(labels == label_id, 1, 0).astype(np.uint8)
        if instance_mask.sum() > 0:
            instance_masks.append(instance_mask)
    
    return instance_masks


def mask_to_multipolygon(
    binary_mask: NDArray[np.uint8], 
    approx_epsilon: float = 0.0
) -> MultiPolygon:
    """Convert a binary mask to MultiPolygon format using contour detection.
    
    Args:
        binary_mask: Binary mask with 0/1 values
        approx_epsilon: Polygon approximation factor (0.0 for no approximation)
        
    Returns:
        MultiPolygon representation of the mask
    """
    # Find contours
    contours, _ = cv2.findContours(
        (binary_mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if approx_epsilon > 0.0 and len(contour) >= 3:
            perimeter = cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
        
        # Convert contour to numpy array and reshape
        contour_array = np.array(contour)
        if contour_array.size == 0:
            continue
        points = contour_array.reshape(-1, 2).astype(float)
        if len(points) >= 3:  # Need at least 3 points for a polygon
            polygon_points = [(float(pt[0]), float(pt[1])) for pt in points]
            polygons.append(polygon_points)
    
    return MultiPolygon(polygons=polygons)


def mask_to_binary_mask_segmentation(binary_mask: NDArray[np.uint8]) -> BinaryMaskSegmentation:
    """Convert a binary mask to BinaryMaskSegmentation format.
    
    Args:
        binary_mask: Binary mask with 0/1 values
        
    Returns:
        BinaryMaskSegmentation representation of the mask
    """
    # Calculate bounding box
    contours, _ = cv2.findContours(
        (binary_mask * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # Empty mask, create minimal bounding box
        bbox = BoundingBox.from_format(bbox=[0.0, 0.0, 1.0, 1.0], format=BoundingBoxFormat.XYWH)
    else:
        # Find bounding box from all contours
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        bbox = BoundingBox.from_format(
            bbox=[float(x), float(y), float(w), float(h)], 
            format=BoundingBoxFormat.XYWH
        )
    
    return BinaryMaskSegmentation.from_binary_mask(
        binary_mask=binary_mask.astype(np.int_), 
        bounding_box=bbox
    )


def match_image_mask_pairs(
    image_glob: str, 
    mask_glob: str, 
    base_path: Path,
    pairing_mode: str = "stem"
) -> List[Tuple[Path, Path]]:
    """Match image files to mask files based on pairing strategy.
    
    Args:
        image_glob: Glob pattern for image files relative to base_path
        mask_glob: Glob pattern for mask files relative to base_path  
        base_path: Base directory for glob patterns
        pairing_mode: Pairing strategy ("stem", "regex", "index")
        
    Returns:
        List of (image_path, mask_path) tuples
        
    Raises:
        ValueError: If pairing mode is invalid or files cannot be matched
    """
    # Find all matching files
    image_paths = sorted(base_path.glob(image_glob))
    mask_paths = sorted(base_path.glob(mask_glob))
    
    if not image_paths:
        raise ValueError(f"No images found matching pattern: {image_glob}")
    if not mask_paths:
        raise ValueError(f"No masks found matching pattern: {mask_glob}")
    
    pairs = []
    
    if pairing_mode == "stem":
        # Match by filename stem (without extension)
        mask_dict = {path.stem: path for path in mask_paths}
        for image_path in image_paths:
            if image_path.stem in mask_dict:
                pairs.append((image_path, mask_dict[image_path.stem]))
    
    elif pairing_mode == "index":
        # Match by sorted index
        min_count = min(len(image_paths), len(mask_paths))
        for i in range(min_count):
            pairs.append((image_paths[i], mask_paths[i]))
    
    elif pairing_mode == "regex":
        # Extract numeric IDs using regex and match by ID
        id_pattern = re.compile(r"(\d+)")
        
        image_dict = {}
        for path in image_paths:
            match = id_pattern.search(path.stem)
            if match:
                image_dict[match.group(1)] = path
        
        mask_dict = {}
        for path in mask_paths:
            match = id_pattern.search(path.stem)
            if match:
                mask_dict[match.group(1)] = path
        
        # Match by common IDs
        for file_id in sorted(image_dict.keys()):
            if file_id in mask_dict:
                pairs.append((image_dict[file_id], mask_dict[file_id]))
    
    else:
        raise ValueError(f"Invalid pairing mode: {pairing_mode}")
    
    if not pairs:
        raise ValueError(f"No matching image/mask pairs found using pairing mode '{pairing_mode}'")
    
    return pairs