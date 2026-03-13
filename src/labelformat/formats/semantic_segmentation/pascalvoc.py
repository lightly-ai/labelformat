"""Pascal VOC semantic segmentation input and output.

Assumptions:
- Masks live under a separate directory mirroring the images directory structure.
- For each image at ``images_dir/<rel>.ext``, the mask is at ``masks_dir/<rel>.png``.
- Masks are PNGs with pixel values equal to class IDs.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageDraw

from labelformat import utils
from labelformat.cli.registry import Task, cli_register
from labelformat.model.binary_mask_segmentation import BinaryMaskSegmentation
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.instance_segmentation import (
    ImageInstanceSegmentation,
    InstanceSegmentationInput,
    InstanceSegmentationOutput,
    SingleInstanceSegmentation,
)
from labelformat.model.multipolygon import MultiPolygon
from labelformat.model.semantic_segmentation import SemanticSegmentationMask

"""TODO(Malte, 11/2025):
Support what is already supported in LightlyTrain. https://docs.lightly.ai/train/stable/semantic_segmentation.html#data
Support using a template against the image filepath. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-a-template-against-the-image-filepath
Support using multi-channel masks. https://docs.lightly.ai/train/stable/semantic_segmentation.html#using-multi-channel-masks
Support optional ignore_classes: list[int] that should be ignored during training. https://docs.lightly.ai/train/stable/semantic_segmentation.html#specify-training-classes
Support merging multiple labels into one class during training. https://docs.lightly.ai/train/stable/semantic_segmentation.html#specify-training-classes
"""


@dataclass
class PascalVOCSemanticSegmentationInput(InstanceSegmentationInput):
    """Pascal VOC semantic segmentation input format."""

    _images_dir: Path
    _masks_dir: Path
    _filename_to_image: dict[str, Image]
    _categories: list[Category]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        # TODO(Michal, 01/2026): Implement when needed.
        raise NotImplementedError()

    @classmethod
    def from_dirs(
        cls,
        images_dir: Path,
        masks_dir: Path,
        class_id_to_name: Mapping[int, str],
    ) -> "PascalVOCSemanticSegmentationInput":
        """Create a PascalVOCSemanticSegmentationInput from directory pairs.

        Args:
            images_dir: Root directory containing images (nested structure allowed).
            masks_dir: Root directory containing PNG masks mirroring images structure.
            class_id_to_name: Mapping of class_id -> class name, with integer keys.

        Raises:
            ValueError: If directories are invalid, a mask is missing or not PNG,
                        or if class_id keys cannot be parsed as integers.
        """
        if not images_dir.is_dir():
            raise ValueError(f"Images directory is not a directory: {images_dir}")
        if not masks_dir.is_dir():
            raise ValueError(f"Masks directory is not a directory: {masks_dir}")

        # Build categories from mapping
        categories = [
            Category(id=cid, name=cname) for cid, cname in class_id_to_name.items()
        ]

        # Collect images using helper and ensure a PNG mask exists for each.
        images_by_filename: dict[str, Image] = {}
        for img in utils.get_images_from_folder(images_dir):
            mask_path = masks_dir / Path(img.filename).with_suffix(".png")
            if not mask_path.is_file():
                raise ValueError(
                    f"Missing mask PNG for image '{img.filename}' at path: {mask_path}"
                )
            images_by_filename[img.filename] = img

        return cls(
            _images_dir=images_dir,
            _masks_dir=masks_dir,
            _filename_to_image=images_by_filename,
            _categories=categories,
        )

    def get_categories(self) -> Iterable[Category]:
        return list(self._categories)

    def get_images(self) -> Iterable[Image]:
        yield from self._filename_to_image.values()

    def get_labels(self) -> Iterable[ImageInstanceSegmentation]:
        """Get semantic segmentation labels.

        Yields an object per image, with one binary mask per category present in the mask.
        The order of objects is sorted by category ID. Reuses the ImageInstanceSegmentation
        as the return type for convenience.
        """
        category_id_to_category = {c.id: c for c in self._categories}
        for image in self.get_images():
            mask = self._get_mask(image_filepath=image.filename)
            category_ids_in_mask = mask.category_ids()
            objects = [
                SingleInstanceSegmentation(
                    category=category_id_to_category[cid],
                    segmentation=mask.to_binary_mask(category_id=cid),
                )
                for cid in sorted(category_ids_in_mask)
            ]
            yield ImageInstanceSegmentation(
                image=image,
                objects=objects,
            )

    def _get_mask(self, image_filepath: str) -> SemanticSegmentationMask:
        # Validate image exists in our index.
        image_obj = self._filename_to_image.get(image_filepath)
        if image_obj is None:
            raise ValueError(
                f"Unknown image filepath {image_filepath}. Use one returned by get_images()."
            )

        mask_path = self._masks_dir / Path(image_filepath).with_suffix(".png")
        if not mask_path.is_file():
            raise ValueError(
                f"Mask PNG not found for image '{image_filepath}': {mask_path}"
            )

        # Load and validate mask by shape and value set.
        with PILImage.open(mask_path) as mimg:
            mask_np: NDArray[np.int_] = np.asarray(mimg, dtype=np.int_)
        _validate_mask(
            image_obj=image_obj,
            mask_np=mask_np,
            valid_class_ids={c.id for c in self._categories},
        )

        return SemanticSegmentationMask.from_array(array=mask_np)


@cli_register(format="pascalvoc", task=Task.INSTANCE_SEGMENTATION)
class PascalVOCSemanticSegmentationOutput(InstanceSegmentationOutput):
    """Pascal VOC semantic segmentation output format.

    Saves one semantic PNG mask per image to
    ``<output_folder>/<masks_folder_name>/...`` and stores the class mapping as JSON in
    ``<output_folder>/<class_map_filename>``.
    """

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-folder",
            type=Path,
            required=True,
            help="Output folder for Pascal VOC semantic segmentation files.",
        )
        parser.add_argument(
            "--masks-folder-name",
            type=str,
            default="SegmentationClass",
            help="Subfolder name where semantic masks are written.",
        )
        parser.add_argument(
            "--class-map-filename",
            type=str,
            default="class_id_to_name.json",
            help="JSON filename for class ID to name mapping.",
        )
        parser.add_argument(
            "--background-class-id",
            type=int,
            default=0,
            help="Class ID used for unlabeled/background pixels.",
        )

    def __init__(
        self,
        output_folder: Path,
        masks_folder_name: str = "SegmentationClass",
        class_map_filename: str = "class_id_to_name.json",
        background_class_id: int = 0,
    ) -> None:
        if background_class_id < 0 or background_class_id > 255:
            raise ValueError(
                "background_class_id must be in [0,255] for Pascal VOC export."
            )

        self._output_folder = output_folder
        self._masks_folder_name = masks_folder_name
        self._class_map_filename = class_map_filename
        self._background_class_id = background_class_id

    def save(self, label_input: InstanceSegmentationInput) -> None:
        category_id_to_name = _get_category_id_to_name(
            categories=label_input.get_categories(),
            background_class_id=self._background_class_id,
        )

        masks_dir = self._output_folder / self._masks_folder_name
        masks_dir.mkdir(parents=True, exist_ok=True)

        for image_label in label_input.get_labels():
            # Initialize an (H, W) mask where every pixel starts as background.
            mask = np.full(
                (image_label.image.height, image_label.image.width),
                fill_value=self._background_class_id,
                dtype=np.int_,
            )
            for obj in image_label.objects:
                if obj.category.id not in category_id_to_name:
                    raise ValueError(
                        f"Category id {obj.category.id} is used in labels but "
                        "missing from categories."
                    )
                obj_mask = _segmentation_to_binary_mask(
                    segmentation=obj.segmentation, image=image_label.image
                )
                mask[obj_mask] = obj.category.id

            mask_path = (masks_dir / image_label.image.filename).with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            _save_mask(mask_path=mask_path, mask=mask)

        class_map_path = self._output_folder / self._class_map_filename
        with class_map_path.open("w") as f:
            json.dump(
                {str(k): v for k, v in sorted(category_id_to_name.items())},
                f,
                indent=2,
            )


def _get_category_id_to_name(
    categories: Iterable[Category], background_class_id: int
) -> dict[int, str]:
    """Build class-id mapping and validate duplicates."""
    category_id_to_name: dict[int, str] = {}
    for category in categories:
        if not 0 <= category.id <= 255:
            raise ValueError(
                "Pascal VOC semantic segmentation export only supports class IDs "
                f"in the range [0, 255]. Got: {category.id}"
            )
        existing_name = category_id_to_name.get(category.id)
        if existing_name is not None and existing_name != category.name:
            raise ValueError(
                "Conflicting names for category id "
                f"{category.id}: '{existing_name}' vs '{category.name}'."
            )
        category_id_to_name[category.id] = category.name

    if background_class_id not in category_id_to_name:
        category_id_to_name[background_class_id] = "background"
    return category_id_to_name


def _segmentation_to_binary_mask(
    segmentation: BinaryMaskSegmentation | MultiPolygon, image: Image
) -> NDArray[np.bool_]:
    if isinstance(segmentation, BinaryMaskSegmentation):
        binary_mask = segmentation.get_binary_mask().astype(np.uint8, copy=False)
    elif isinstance(segmentation, MultiPolygon):
        binary_mask = _multipolygon_to_binary_mask(
            multipolygon=segmentation,
            width=image.width,
            height=image.height,
        )
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    expected_shape = (image.height, image.width)
    if binary_mask.shape != expected_shape:
        raise ValueError(
            f"Segmentation mask shape must match image dimensions for "
            f"'{image.filename}': got {binary_mask.shape}, expected {expected_shape}."
        )
    return binary_mask > 0


def _multipolygon_to_binary_mask(
    multipolygon: MultiPolygon, width: int, height: int
) -> NDArray[np.uint8]:
    mask_img = PILImage.new(mode="L", size=(width, height), color=0)
    draw = ImageDraw.Draw(mask_img)
    for polygon in multipolygon.polygons:
        if len(polygon) < 3:
            raise ValueError(
                f"Polygon must contain at least 3 points, got {len(polygon)}."
            )
        draw.polygon(xy=polygon, fill=1, outline=1)
    return np.asarray(mask_img, dtype=np.uint8)


def _pascal_voc_palette() -> list[int]:
    """Build the standard Pascal VOC palette (256 colors, RGB triples)."""
    palette = [0] * (256 * 3)
    for class_id in range(256):
        label = class_id
        red = 0
        green = 0
        blue = 0
        bit_index = 0
        while label:
            red |= ((label >> 0) & 1) << (7 - bit_index)
            green |= ((label >> 1) & 1) << (7 - bit_index)
            blue |= ((label >> 2) & 1) << (7 - bit_index)
            bit_index += 1
            label >>= 3
        palette[(class_id * 3) : (class_id * 3 + 3)] = [red, green, blue]
    return palette


_PASCAL_VOC_PALETTE = _pascal_voc_palette()


def _save_mask(mask_path: Path, mask: NDArray[np.int_]) -> None:
    mask_img = PILImage.fromarray(mask.astype(np.uint8), mode="P")
    mask_img.putpalette(_PASCAL_VOC_PALETTE)
    mask_img.save(mask_path)


def _validate_mask(
    image_obj: Image, mask_np: NDArray[np.int_], valid_class_ids: set[int]
) -> None:
    """Validate a semantic segmentation mask against an image and categories.

    Args:
        image_obj: The image metadata with filename, width, and height used for shape validation.
        mask_np: The mask as a 2D numpy array with integer class IDs.
        valid_class_ids: The set of allowed class IDs that may appear in the mask.

    Raises:
        ValueError: If the mask is not 2D, does not match the image size, or contains
            class IDs not present in `valid_class_ids`.
    """
    if mask_np.ndim != 2:
        raise ValueError(
            f"Mask must be 2D (H, W) for: {image_obj.filename}. Got shape {mask_np.shape}"
        )

    mh, mw = int(mask_np.shape[0]), int(mask_np.shape[1])
    if (mw, mh) != (image_obj.width, image_obj.height):
        raise ValueError(
            f"Mask shape must match image dimensions for '{image_obj.filename}': "
            f"mask (W,H)=({mw},{mh}) vs image (W,H)=({image_obj.width},{image_obj.height})"
        )

    uniques = np.unique(mask_np.astype(int))
    unique_values = set(uniques)
    unknown_values = unique_values.difference(valid_class_ids)
    if unknown_values:
        raise ValueError(
            f"Mask contains unknown class ids: {', '.join(map(str, sorted(unknown_values)))}"
        )
