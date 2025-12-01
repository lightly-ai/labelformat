from pathlib import Path
from typing import Tuple

import PIL.Image
import pytest

from labelformat.utils import (
    ImageDimensionError,
    get_image_dimensions,
    get_jpeg_dimensions,
    get_png_dimensions,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_get_jpeg_dimensions() -> None:
    image_path = FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000109005.jpg"
    width, height = get_jpeg_dimensions(image_path)
    assert width == 640
    assert height == 428


def test_get_jpeg_dimensions__baseline(tmp_path: Path) -> None:
    # Tests SOF0 (0xC0) - Baseline DCT.
    jpeg_path = tmp_path / "baseline.jpg"
    _create_test_jpeg(path=jpeg_path, size=(800, 600))

    width, height = get_jpeg_dimensions(jpeg_path)
    assert width == 800
    assert height == 600


def test_get_jpeg_dimensions__progressive(tmp_path: Path) -> None:
    # Tests SOF2 (0xC2) - Progressive DCT with DHT markers before SOF.
    jpeg_path = tmp_path / "progressive.jpg"
    _create_test_jpeg(path=jpeg_path, size=(1920, 1440), progressive=True)

    width, height = get_jpeg_dimensions(jpeg_path)
    assert width == 1920
    assert height == 1440


def test_get_jpeg_dimensions__optimized(tmp_path: Path) -> None:
    # Tests SOF0 (0xC0) with custom Huffman tables (more DHT markers before SOF).
    jpeg_path = tmp_path / "optimized.jpg"
    _create_test_jpeg(path=jpeg_path, size=(1024, 768), optimize=True)

    width, height = get_jpeg_dimensions(jpeg_path)
    assert width == 1024
    assert height == 768


def test_get_jpeg_dimensions__progressive_optimized(tmp_path: Path) -> None:
    # Tests SOF2 (0xC2) with custom Huffman tables.
    jpeg_path = tmp_path / "progressive_optimized.jpg"
    _create_test_jpeg(
        path=jpeg_path, size=(2048, 1536), progressive=True, optimize=True
    )

    width, height = get_jpeg_dimensions(jpeg_path)
    assert width == 2048
    assert height == 1536


def test_get_jpeg_dimensions__nonexistent_file() -> None:
    with pytest.raises(ImageDimensionError):
        get_jpeg_dimensions(Path("nonexistent.jpg"))


def test_get_jpeg_dimensions__invalid_format() -> None:
    yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
    with pytest.raises(ImageDimensionError):
        get_jpeg_dimensions(yaml_file)


def test_get_png_dimensions() -> None:
    png_path = FIXTURES_DIR / "image_file_loading/0001.png"
    width, height = get_png_dimensions(png_path)
    assert width == 278
    assert height == 181


def test_get_png_dimensions__nonexistent_file() -> None:
    with pytest.raises(ImageDimensionError):
        get_png_dimensions(Path("nonexistent.png"))


def test_get_png_dimensions__invalid_format() -> None:
    yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
    with pytest.raises(ImageDimensionError):
        get_png_dimensions(yaml_file)


def test_get_image_dimensions__jpeg() -> None:
    jpeg_path = FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000109005.jpg"
    width, height = get_image_dimensions(jpeg_path)
    assert width == 640
    assert height == 428


def test_get_image_dimensions__jpeg_second_file() -> None:
    jpeg_path = FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000036086.jpg"
    width, height = get_image_dimensions(jpeg_path)
    assert width == 482
    assert height == 640


def test_get_image_dimensions__png() -> None:
    png_path = FIXTURES_DIR / "image_file_loading/0001.png"
    width, height = get_image_dimensions(png_path)
    assert width == 278
    assert height == 181


def test_get_image_dimensions__unsupported_format() -> None:
    yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
    with pytest.raises(Exception):
        get_image_dimensions(yaml_file)


def _create_test_jpeg(
    path: Path,
    size: Tuple[int, int],
    progressive: bool = False,
    optimize: bool = False,
) -> None:
    img = PIL.Image.new("RGB", size, color="red")
    img.save(path, "JPEG", progressive=progressive, optimize=optimize)
