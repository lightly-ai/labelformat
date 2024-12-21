from pathlib import Path

import pytest
import yaml

from labelformat.formats.yolov8 import _YOLOv8BaseInput
from labelformat.model.category import Category


@pytest.fixture
def expected_categories():
    return [
        Category(id=0, name="person"),
        Category(id=1, name="dog"),
        Category(id=2, name="cat"),
    ]


def test_get_categories_dict_format(tmp_path: Path, expected_categories) -> None:
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


def test_get_categories_list_format(tmp_path: Path, expected_categories) -> None:
    config = {"path": ".", "train": "images", "names": ["person", "dog", "cat"]}
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)

    input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
    categories = list(input_obj.get_categories())
    assert categories == expected_categories


def test_get_categories_yaml_block_format(tmp_path: Path, expected_categories) -> None:
    config = """
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
    with pytest.raises(ValueError) as exc_info:
        list(input_obj.get_categories())

    assert "Invalid 'names' format" in str(exc_info.value)
    assert "Expected dictionary or list" in str(exc_info.value)
