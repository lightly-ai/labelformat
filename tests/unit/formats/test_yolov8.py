from pathlib import Path

import pytest
import yaml

from labelformat.formats.yolov8 import _YOLOv8BaseInput
from labelformat.model.category import Category


def test_get_categories_supports_list_and_dict_format(tmp_path: Path) -> None:
    # Test dictionary format (original)
    dict_config = {
        "path": ".",
        "train": "images",
        "names": {0: "person", 1: "dog", 2: "cat"},
    }
    dict_file = tmp_path / "dict.yaml"
    with dict_file.open("w") as f:
        yaml.safe_dump(dict_config, f)

    dict_input = _YOLOv8BaseInput(input_file=dict_file, input_split="train")
    dict_categories = list(dict_input.get_categories())

    # Test list format with explicit brackets
    list_config = {"path": ".", "train": "images", "names": ["person", "dog", "cat"]}
    list_file = tmp_path / "list.yaml"
    with list_file.open("w") as f:
        yaml.safe_dump(list_config, f)

    list_input = _YOLOv8BaseInput(input_file=list_file, input_split="train")
    list_categories = list(list_input.get_categories())

    # Test list format with YAML block sequence
    block_config = """
    path: .
    train: images
    names:
      - person
      - dog
      - cat
    """
    block_file = tmp_path / "block.yaml"
    with block_file.open("w") as f:
        f.write(block_config)

    block_input = _YOLOv8BaseInput(input_file=block_file, input_split="train")
    block_categories = list(block_input.get_categories())

    # All formats should produce the same categories
    expected = [
        Category(id=0, name="person"),
        Category(id=1, name="dog"),
        Category(id=2, name="cat"),
    ]
    assert dict_categories == expected
    assert list_categories == expected
    assert block_categories == expected
