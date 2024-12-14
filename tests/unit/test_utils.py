from pathlib import Path

import pytest

from labelformat.utils import (
    ImageDimensionError,
    get_image_dimensions,
    get_jpeg_dimensions,
    get_png_dimensions,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestImageDimensions:
    def test_jpeg_dimensions_valid_file(self) -> None:
        image_path = (
            FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000109005.jpg"
        )
        width, height = get_jpeg_dimensions(image_path)
        assert width == 640
        assert height == 428

    def test_jpeg_dimensions_nonexistent_file(self) -> None:
        with pytest.raises(ImageDimensionError):
            get_jpeg_dimensions(Path("nonexistent.jpg"))

    def test_jpeg_dimensions_invalid_format(self) -> None:
        yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
        with pytest.raises(ImageDimensionError):
            get_jpeg_dimensions(yaml_file)

    def test_png_dimensions_valid_file(self) -> None:
        png_path = FIXTURES_DIR / "image_file_loading/0001.png"
        width, height = get_png_dimensions(png_path)
        assert width == 278
        assert height == 181

    def test_png_dimensions_nonexistent_file(self) -> None:
        with pytest.raises(ImageDimensionError):
            get_png_dimensions(Path("nonexistent.png"))

    def test_png_dimensions_invalid_format(self) -> None:
        yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
        with pytest.raises(ImageDimensionError):
            get_png_dimensions(yaml_file)

    def test_get_image_dimensions_jpeg_first_file(self) -> None:
        jpeg_path = (
            FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000109005.jpg"
        )
        width, height = get_image_dimensions(jpeg_path)
        assert width == 640
        assert height == 428

    def test_get_image_dimensions_jpeg_second_file(self) -> None:
        jpeg_path = (
            FIXTURES_DIR / "instance_segmentation/YOLOv8/images/000000036086.jpg"
        )
        width, height = get_image_dimensions(jpeg_path)
        assert width == 482
        assert height == 640

    def test_get_image_dimensions_png(self) -> None:
        png_path = FIXTURES_DIR / "image_file_loading/0001.png"
        width, height = get_image_dimensions(png_path)
        assert width == 278
        assert height == 181

    def test_get_image_dimensions_unsupported_format(self) -> None:
        yaml_file = FIXTURES_DIR / "object_detection/YOLOv8/example.yaml"
        with pytest.raises(Exception):
            get_image_dimensions(yaml_file)
