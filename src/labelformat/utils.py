import logging
from pathlib import Path
from typing import Iterable

import PIL.Image

from labelformat.model.image import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


class ImageDimensionError(Exception):
    """Raised when unable to extract image dimensions using fast methods."""

    pass


def get_jpeg_dimensions(file_path: Path) -> tuple[int, int]:
    """Try to efficiently get JPEG dimensions from file headers without decoding the image.

    This method reads only the JPEG file headers looking for the Start Of Frame (SOFn)
    marker which contains the dimensions. This is much faster than decoding the entire
    image as it:
    - Only reads the file headers (typically a few KB) instead of the entire file
    - Doesn't perform any image decompression
    - Doesn't load the pixel data into memory

    This works for most standard JPEG files (including progressive JPEGs) but may fail
    for some unusual formats or corrupted files. In those cases, an ImageDimensionError
    is raised and a full image decode may be needed as fallback.

    Args:
        file_path: Path to the JPEG file

    Returns:
        Tuple of (width, height)

    Raises:
        ImageDimensionError: If dimensions cannot be extracted from headers
    """
    try:
        with open(file_path, "rb") as img_file:
            # Skip SOI marker
            img_file.seek(2)
            while True:
                marker = img_file.read(2)
                if len(marker) < 2:
                    raise ImageDimensionError("Invalid JPEG format")
                # Find SOFn marker
                if 0xFF == marker[0] and marker[1] in range(0xC0, 0xCF):
                    # Skip marker length
                    img_file.seek(3, 1)
                    h = int.from_bytes(img_file.read(2), "big")
                    w = int.from_bytes(img_file.read(2), "big")
                    return w, h
                # Skip to next marker
                length = int.from_bytes(img_file.read(2), "big")
                img_file.seek(length - 2, 1)
    except Exception as e:
        raise ImageDimensionError(f"Failed to read JPEG dimensions: {str(e)}")


def get_png_dimensions(file_path: Path) -> tuple[int, int]:
    """Try to efficiently get PNG dimensions from file headers without decoding the image.

    This method reads only the PNG IHDR (Image Header) chunk which is always the first
    chunk after the PNG signature. This is much faster than decoding the entire image as it:
    - Only reads the first ~30 bytes of the file
    - Doesn't perform any image decompression
    - Doesn't load the pixel data into memory

    This works for all valid PNG files since the IHDR chunk is mandatory and must appear
    first according to the PNG specification. However, it may fail for corrupted files
    or files that don't follow the PNG spec. In those cases, an ImageDimensionError is
    raised and a full image decode may be needed as fallback.

    Args:
        file_path: Path to the PNG file

    Returns:
        Tuple of (width, height)

    Raises:
        ImageDimensionError: If dimensions cannot be extracted from headers
    """
    try:
        with open(file_path, "rb") as img_file:
            # Skip PNG signature
            img_file.seek(8)
            # Read IHDR chunk
            chunk_length = int.from_bytes(img_file.read(4), "big")
            chunk_type = img_file.read(4)
            if chunk_type == b"IHDR":
                w = int.from_bytes(img_file.read(4), "big")
                h = int.from_bytes(img_file.read(4), "big")
                return w, h
            raise ImageDimensionError("Invalid PNG format")
    except Exception as e:
        raise ImageDimensionError(f"Failed to read PNG dimensions: {str(e)}")


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image dimensions using the most efficient method available.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)

    Raises:
        Exception: If image dimensions cannot be extracted using any method
    """
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        try:
            return get_jpeg_dimensions(image_path)
        except ImageDimensionError:
            pass
    elif suffix == ".png":
        try:
            return get_png_dimensions(image_path)
        except ImageDimensionError:
            pass

    with PIL.Image.open(image_path) as img:
        return img.size


def get_images_from_folder(folder: Path) -> Iterable[Image]:
    """Yields an Image structure for all images in the given folder.

    The order of the images is arbitrary. Images in nested folders are included.

    Args:
        folder: Path to the folder containing images.
    """
    image_id = 0
    logger.debug(f"Listing images in '{folder}'...")
    for image_path in folder.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.debug(f"Skipping non-image file '{image_path}'")
            continue
        image_filename = str(image_path.relative_to(folder))
        image_width, image_height = get_image_dimensions(image_path)
        yield Image(
            id=image_id,
            filename=image_filename,
            width=image_width,
            height=image_height,
        )
        image_id += 1
