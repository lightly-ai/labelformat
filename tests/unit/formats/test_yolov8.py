from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import pytest
import yaml

from labelformat.formats.yolov8 import _YOLOv8BaseInput
from labelformat.model.category import Category


@pytest.fixture
def expected_categories() -> List[Category]:
    return [
        Category(id=0, name="person"),
        Category(id=1, name="dog"),
        Category(id=2, name="cat"),
    ]


def test_get_categories_dict_format(
    tmp_path: Path, expected_categories: List[Category]
) -> None:
    config = {
        "path": ".",
        "train": "images",
        "names": {0: "person", 1: "dog", 2: "cat"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    categories = list(input_obj.get_categories())
    assert categories == expected_categories


def test_get_categories_list_format(
    tmp_path: Path, expected_categories: List[Category]
) -> None:
    config = {
        "path": ".",
        "train": "images",
        "names": ["person", "dog", "cat"],
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    categories = list(input_obj.get_categories())
    assert categories == expected_categories


def test_get_categories_yaml_block_format(
    tmp_path: Path, expected_categories: List[Category]
) -> None:
    config: str = """
    path: .
    train: images
    names:
      - person
      - dog
      - cat
    """
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        f.write(config)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    categories = list(input_obj.get_categories())
    assert categories == expected_categories


def test_root_dir_with_explicit_path(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    config = {"path": ".", "train": "images", "names": ["person"]}
    config_file = dataset_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    assert input_obj._root_dir() == dataset_dir


def test_root_dir_without_path(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    config = {"train": "images", "names": ["person"]}
    config_file = dataset_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    assert input_obj._root_dir() == dataset_dir


def test_invalid_names_format(tmp_path: Path) -> None:
    config = {
        "path": ".",
        "train": "images",
        "names": 123,  # Invalid format
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    with pytest.raises(TypeError):  # Will fail when trying to use len() on int
        list(input_obj.get_categories())


def test_labels_dir_relative_to_path(tmp_path: Path) -> None:
    """Test labels directory resolution for paths relative to dataset root."""
    config = {
        "path": "../datasets/coco8",
        "train": "images/train",
        "names": ["person"],
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    expected = tmp_path / "../datasets/coco8/labels/train"
    assert input_obj._labels_dir() == expected


def test_labels_dir_absolute_path(tmp_path: Path) -> None:
    """Test labels directory resolution for absolute paths."""
    config = {
        "path": ".",
        "train": "../train/images",
        "names": ["head", "helmet", "person"],
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    expected = tmp_path / "../train/labels"
    assert input_obj._labels_dir() == expected


def test_labels_dir_with_images_in_path(tmp_path: Path) -> None:
    """Test labels directory resolution when 'images' appears in the root path."""
    config = {
        "path": "mydataset/images/dataset1",
        "train": "images/train",
        "names": ["person"],
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    expected = tmp_path / "mydataset/images/dataset1/labels/train"
    assert input_obj._labels_dir() == expected


def test_labels_dir_without_path(tmp_path: Path) -> None:
    """Test labels directory resolution when 'path' is not specified in config."""
    config = {
        "train": "images/train",
        "names": ["person"],
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    expected = tmp_path / "labels/train"
    assert input_obj._labels_dir() == expected


def test_multilevel_relative_paths_with_dot(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for split in ["train", "valid", "test"]:
        (dataset_root / split / "images").mkdir(parents=True)
        (dataset_root / split / "labels").mkdir(parents=True)

    config = {
        "train": "./train/images",
        "valid": "./valid/images",
        "test": "./test/images",
        "names": ["person"],
    }
    config_file = dataset_root / "data.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")

    assert input_obj._root_dir() == dataset_root
    assert input_obj._images_dir() == dataset_root / "train" / "images"
    assert input_obj._labels_dir() == dataset_root / "train" / "labels"


def test_multilevel_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for split in ["train", "valid", "test"]:
        (dataset_root / split / "images").mkdir(parents=True)
        (dataset_root / split / "labels").mkdir(parents=True)

    config = {
        "train": "../train/images",
        "valid": "../valid/images",
        "test": "../test/images",
        "names": ["person"],
    }
    config_file = dataset_root / "data.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")

    assert input_obj._root_dir() == dataset_root
    assert input_obj._images_dir() == dataset_root / "train" / "images"
    assert input_obj._labels_dir() == dataset_root / "train" / "labels"


def test_intentional_parent_dir_reference(tmp_path: Path) -> None:
    """Test that paths with ../ are preserved when the directory exists."""
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()

    dataset_dir = parent_dir / "dataset"
    dataset_dir.mkdir()

    target_images_dir = parent_dir / "images"
    target_images_dir.mkdir()

    target_labels_dir = parent_dir / "labels"
    target_labels_dir.mkdir()

    config = {
        "path": ".",
        "train": "../images",  # This is intentionally using ../ to go up to parent/images
        "names": ["person"],
    }
    config_file = dataset_dir / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")

    expected_images_dir = (dataset_dir / "../images").resolve()
    assert input_obj._images_dir().resolve() == expected_images_dir

    expected_labels_dir = (dataset_dir / "../labels").resolve()
    assert input_obj._labels_dir().resolve() == expected_labels_dir
