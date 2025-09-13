"""Mask utilities implemented with Pillow and NumPy, without OpenCV."""

from pathlib import Path
from typing import Generator, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.multipolygon import MultiPolygon

# Copied from OpenCV and then vibe-translated to using only numpy.
# --- I/O and thresholding ---


def _read_grayscale(mask_path: Path) -> NDArray[np.uint8]:
    img = Image.open(mask_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _otsu_threshold(img: NDArray[np.uint8]) -> int:
    hist: NDArray[np.float32] = np.bincount(img.ravel(), minlength=256).astype(
        np.float32
    )
    total = img.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan

    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    return int(np.nanargmax(sigma_b2))


def _apply_threshold(
    img: NDArray[np.uint8], threshold: Union[int, None]
) -> NDArray[np.uint8]:
    t = _otsu_threshold(img) if threshold is None else int(threshold)
    return (img > t).astype(np.uint8)


def _ensure_odd(n: int) -> int:
    if n <= 0:
        return 0
    return n if n % 2 == 1 else n + 1


def _morph_open_close(
    binary: NDArray[np.uint8], morph_open: int, morph_close: int
) -> NDArray[np.uint8]:
    if morph_open <= 0 and morph_close <= 0:
        return binary

    img = Image.fromarray((binary * 255).astype(np.uint8))

    if morph_open > 0:
        k = _ensure_odd(morph_open)
        img = img.filter(ImageFilter.MinFilter(size=k)).filter(
            ImageFilter.MaxFilter(size=k)
        )

    if morph_close > 0:
        k = _ensure_odd(morph_close)
        img = img.filter(ImageFilter.MaxFilter(size=k)).filter(
            ImageFilter.MinFilter(size=k)
        )

    return (np.asarray(img) > 0).astype(np.uint8)


def binarize_mask(
    mask_path: Path,
    threshold: Union[int, None] = None,
    morph_open: int = 0,
    morph_close: int = 0,
) -> NDArray[np.uint8]:
    """Read an image, produce a 0 or 1 mask, and apply optional morphology."""
    try:
        img = _read_grayscale(mask_path=mask_path)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to read mask image: {mask_path}") from e

    binary = _apply_threshold(img=img, threshold=threshold)
    return _morph_open_close(
        binary=binary, morph_open=morph_open, morph_close=morph_close
    )


# --- Connected components (8-connectivity) ---


def _uf_find(par: List[int], x: int) -> int:
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def _uf_union(par: List[int], a: int, b: int) -> None:
    ra, rb = _uf_find(par, a), _uf_find(par, b)
    if ra != rb:
        par[rb] = ra


def _connected_components(
    binary_mask: NDArray[np.uint8], connectivity: int = 8
) -> Tuple[int, NDArray[np.int32]]:
    h, w = binary_mask.shape
    labels: NDArray[np.int32] = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    parent: List[int] = [0]

    def neighbors(y: int, x: int) -> Generator[Tuple[int, int], None, None]:
        offs = (
            [(-1, 0), (-1, -1), (-1, 1), (0, -1)]
            if connectivity == 8
            else [(-1, 0), (0, -1)]
        )
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for y in range(h):
        for x in range(w):
            if binary_mask[y, x] == 0:
                continue
            neigh_labels = [
                labels[ny, nx] for ny, nx in neighbors(y, x) if labels[ny, nx] != 0
            ]
            if not neigh_labels:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                m = min(neigh_labels)
                labels[y, x] = m
                for nl in neigh_labels:
                    if nl != m:
                        _uf_union(parent, m, nl)

    label_map = {0: 0}
    new_label = 1
    for y in range(h):
        for x in range(w):
            l = labels[y, x]
            if l == 0:
                continue
            root = _uf_find(parent, l)
            if root not in label_map:
                label_map[root] = new_label
                new_label += 1
            labels[y, x] = label_map[root]

    return new_label, labels


def extract_instance_masks(binary_mask: NDArray[np.uint8]) -> List[NDArray[np.uint8]]:
    """Split a binary mask into 8-connected components."""
    _, labels = _connected_components(binary_mask=binary_mask, connectivity=8)
    max_label = int(labels.max(initial=0))
    out: List[NDArray[np.uint8]] = []
    for lid in range(1, max_label + 1):
        inst = (labels == lid).astype(np.uint8)
        if inst.any():
            out.append(inst)
    return out


# --- Contour tracing for polygons (preserves concavities) ---


def _rdp(points: NDArray[np.float32], epsilon: float) -> NDArray[np.float32]:
    if len(points) <= 3 or epsilon <= 0:
        return points

    def _point_line_dist(
        p: NDArray[np.float32], a: NDArray[np.float32], b: NDArray[np.float32]
    ) -> float:
        if np.allclose(a, b):
            return float(np.linalg.norm(p - a))
        return float(abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a))

    def _rec(pts: NDArray[np.float32]) -> NDArray[np.float32]:
        a, b = pts[0], pts[-1]
        dmax = 0.0
        idx = 0
        for i in range(1, len(pts) - 1):
            d = _point_line_dist(pts[i], a, b)
            if d > dmax:
                idx = i
                dmax = d
        if dmax > epsilon:
            left = _rec(pts[: idx + 1])
            right = _rec(pts[idx:])
            return np.concatenate([left[:-1], right], axis=0)
        return np.stack([a, b], axis=0)

    return _rec(points)


def _find_boundary_pixels(binary_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    m: NDArray[np.uint8] = binary_mask.astype(np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    c = p[1:-1, 1:-1]
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    boundary = (c == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    result: NDArray[np.uint8] = boundary.astype(np.uint8)
    return result


def _trace_outer_contour(binary_mask: NDArray[np.uint8]) -> List[Tuple[float, float]]:
    h, w = binary_mask.shape
    if binary_mask.sum() == 0:
        return []

    boundary = _find_boundary_pixels(binary_mask)
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return []

    start_idx = int(np.lexsort((xs, ys))[0])
    sy, sx = int(ys[start_idx]), int(xs[start_idx])

    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def is_boundary(y: int, x: int) -> bool:
        return 0 <= y < h and 0 <= x < w and boundary[y, x] == 1

    contour: List[Tuple[float, float]] = []
    y, x = sy, sx
    prev_dir = 6
    visited_first = False

    max_steps = int(10 * boundary.sum()) + 10
    steps = 0

    while steps < max_steps:
        contour.append((float(x), float(y)))
        steps += 1

        found_next = False
        for k in range(8):
            di = (prev_dir + 1 + k) % 8
            dy, dx = dirs[di]
            ny, nx = y + dy, x + dx
            if is_boundary(ny, nx):
                y, x = ny, nx
                prev_dir = (di + 4) % 8
                found_next = True
                break

        if not found_next:
            break
        if y == sy and x == sx:
            if visited_first:
                break
            visited_first = True

    return contour


def mask_to_multipolygon(
    binary_mask: NDArray[np.uint8], approx_epsilon: float = 0.0
) -> MultiPolygon:
    """Trace the outer contour of a single instance mask and return a polygon.

    If you need holes, use RLE based segmentation instead.
    """
    contour = _trace_outer_contour(binary_mask)
    if len(contour) < 3:
        return MultiPolygon(polygons=[])

    pts: NDArray[np.float32] = np.array(contour, dtype=np.float32)

    if approx_epsilon > 0.0 and len(pts) >= 3:
        diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        per = float(np.sqrt((diffs**2).sum(axis=1)).sum())
        pts = _rdp(pts, epsilon=approx_epsilon * max(per, 1.0))

    polygon_points = [(float(x), float(y)) for x, y in pts]
    return MultiPolygon(polygons=[polygon_points])


def mask_to_binary_mask_segmentation(
    binary_mask: NDArray[np.uint8],
) -> BinaryMaskSegmentation:
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        bbox = BoundingBox.from_format(
            bbox=[0.0, 0.0, 1.0, 1.0], format=BoundingBoxFormat.XYWH
        )
    else:
        x0 = float(xs.min())
        y0 = float(ys.min())
        x1 = float(xs.max())
        y1 = float(ys.max())
        w = float(x1 - x0 + 1.0)
        h = float(y1 - y0 + 1.0)
        bbox = BoundingBox.from_format(
            bbox=[x0, y0, w, h], format=BoundingBoxFormat.XYWH
        )

    return BinaryMaskSegmentation.from_binary_mask(
        binary_mask=binary_mask.astype(np.int_),
        bounding_box=bbox,
    )


def match_image_mask_pairs(
    image_glob: str,
    mask_glob: str,
    base_path: Path,
    pairing_mode: str = "stem",
) -> List[Tuple[Path, Path]]:
    """Match image files to mask files using a pairing strategy."""
    import re

    image_paths = sorted(base_path.glob(image_glob))
    mask_paths = sorted(base_path.glob(mask_glob))

    if not image_paths:
        raise ValueError(f"No images found matching pattern: {image_glob}")
    if not mask_paths:
        raise ValueError(f"No masks found matching pattern: {mask_glob}")

    pairs: List[Tuple[Path, Path]] = []

    if pairing_mode == "stem":
        mask_dict = {path.stem: path for path in mask_paths}
        for image_path in image_paths:
            if image_path.stem in mask_dict:
                pairs.append((image_path, mask_dict[image_path.stem]))

    elif pairing_mode == "index":
        min_count = min(len(image_paths), len(mask_paths))
        for i in range(min_count):
            pairs.append((image_paths[i], mask_paths[i]))

    elif pairing_mode == "regex":
        id_pattern = re.compile(r"(\d+)")
        image_dict = {}
        for path in image_paths:
            m = id_pattern.search(path.stem)
            if m:
                image_dict[m.group(1)] = path
        mask_dict = {}
        for path in mask_paths:
            m = id_pattern.search(path.stem)
            if m:
                mask_dict[m.group(1)] = path
        for file_id in sorted(image_dict.keys()):
            if file_id in mask_dict:
                pairs.append((image_dict[file_id], mask_dict[file_id]))

    else:
        raise ValueError(f"Invalid pairing mode: {pairing_mode}")

    if not pairs:
        raise ValueError(
            f"No matching image/mask pairs found using pairing mode '{pairing_mode}'"
        )

    return pairs
