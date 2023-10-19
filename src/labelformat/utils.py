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
        image_width, image_height = PIL.Image.open(image_path).size
        yield Image(
            id=image_id,
            filename=image_filename,
            width=image_width,
            height=image_height,
        )
        image_id += 1
