from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, TypedDict, Union

import pytest
import yaml

from labelformat.formats.yolov8 import _YOLOv8BaseInput
from labelformat.model.category import Category


class YOLOv8Config(TypedDict, total=False):
    path: str
    train: str
    valid: str
    test: str
    nc: int
    names: Union[Dict[int, str], List[str]]


@pytest.fixture
def expected_categories() -> List[Category]:
    return [
        Category(id=0, name="person"),
        Category(id=1, name="dog"),
        Category(id=2, name="cat"),
    ]


@pytest.fixture
def config_file_factory(tmp_path: Path) -> Callable[[Union[YOLOv8Config, str]], Path]:
    """Factory fixture to create config files with different formats."""

    def _create_config(config_data: Union[YOLOv8Config, str]) -> Path:
        config_file = tmp_path / "config.yaml"

        if isinstance(config_data, str):
            with config_file.open("w") as f:
                f.write(config_data)
        else:
            with config_file.open("w") as f:
                yaml.safe_dump(config_data, f)

        return config_file

    return _create_config


class Test_YOLOv8BaseInput:
    class Test_GetCategories:
        def test_extracts_categories_from_dict_format(
            self,
            config_file_factory: Callable[[Union[YOLOv8Config, str]], Path],
            expected_categories: List[Category],
        ) -> None:
            config: YOLOv8Config = {
                "path": ".",
                "train": "images",
                "names": {0: "person", 1: "dog", 2: "cat"},
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            categories = list(input_obj.get_categories())
            assert categories == expected_categories

        def test_extracts_categories_from_list_format(
            self,
            config_file_factory: Callable[[Union[YOLOv8Config, str]], Path],
            expected_categories: List[Category],
        ) -> None:
            config: YOLOv8Config = {
                "path": ".",
                "train": "images",
                "names": ["person", "dog", "cat"],
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            categories = list(input_obj.get_categories())
            assert categories == expected_categories

        def test_extracts_categories_from_yaml_block_format(
            self,
            config_file_factory: Callable[[Union[YOLOv8Config, str]], Path],
            expected_categories: List[Category],
        ) -> None:
            config = """
            path: .
            train: images
            names:
              - person
              - dog
              - cat
            """
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            categories = list(input_obj.get_categories())
            assert categories == expected_categories

        def test_raises_error_for_invalid_names_format(
            self, config_file_factory: Callable[[Union[YOLOv8Config, str]], Path]
        ) -> None:
            config = {
                "path": ".",
                "train": "images",
                "names": 123,  # Invalid format will make mypy raise an error
            }
            config_file = config_file_factory(config)  # type: ignore[arg-type]

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            with pytest.raises(TypeError):  # Will fail when trying to use len() on int
                list(input_obj.get_categories())

    class Test_RootDir:
        def test_resolves_root_dir_with_explicit_path(self, tmp_path: Path) -> None:
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()

            config = {"path": ".", "train": "images", "names": ["person"]}
            config_file = dataset_dir / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config, f)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            assert input_obj._root_dir() == dataset_dir

        def test_resolves_root_dir_without_path(self, tmp_path: Path) -> None:
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()

            config = {"train": "images", "names": ["person"]}
            config_file = dataset_dir / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config, f)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            assert input_obj._root_dir() == dataset_dir

    class Test_LabelsDir:
        def test_resolves_labels_dir_relative_to_path(
            self, config_file_factory: Callable[[Union[YOLOv8Config, str]], Path]
        ) -> None:
            config: YOLOv8Config = {
                "path": "../datasets/coco8",
                "train": "images/train",
                "names": ["person"],
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            expected = config_file.parent / "../datasets/coco8/labels/train"
            assert input_obj._labels_dir() == expected

        def test_resolves_labels_dir_for_absolute_path(
            self, config_file_factory: Callable[[Union[YOLOv8Config, str]], Path]
        ) -> None:
            config: YOLOv8Config = {
                "path": ".",
                "train": "../train/images",
                "names": ["head", "helmet", "person"],
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            expected = config_file.parent / "../train/labels"
            assert input_obj._labels_dir() == expected

        def test_resolves_labels_dir_with_images_in_path(
            self, config_file_factory: Callable[[Union[YOLOv8Config, str]], Path]
        ) -> None:
            config: YOLOv8Config = {
                "path": "mydataset/images/dataset1",
                "train": "images/train",
                "names": ["person"],
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            expected = config_file.parent / "mydataset/images/dataset1/labels/train"
            assert input_obj._labels_dir() == expected

        def test_resolves_labels_dir_without_path(
            self, config_file_factory: Callable[[Union[YOLOv8Config, str]], Path]
        ) -> None:
            config: YOLOv8Config = {
                "train": "images/train",
                "names": ["person"],
            }
            config_file = config_file_factory(config)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")
            expected = config_file.parent / "labels/train"
            assert input_obj._labels_dir() == expected

    class Test_MultilevelPaths:
        def test_handles_relative_paths_with_dot_notation(self, tmp_path: Path) -> None:
            dataset_root = tmp_path / "dataset"
            for split in ["train", "valid", "test"]:
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir(parents=True)

            config: YOLOv8Config = {
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

        def test_handles_relative_paths(self, tmp_path: Path) -> None:
            dataset_root = tmp_path / "dataset"
            for split in ["train", "valid", "test"]:
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir(parents=True)

            config: YOLOv8Config = {
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

        def test_preserves_parent_dir_references_when_directory_exists(
            self, tmp_path: Path
        ) -> None:
            parent_dir = tmp_path / "parent"
            parent_dir.mkdir()

            dataset_dir = parent_dir / "dataset"
            dataset_dir.mkdir()

            target_images_dir = parent_dir / "images"
            target_images_dir.mkdir()

            target_labels_dir = parent_dir / "labels"
            target_labels_dir.mkdir()

            config: YOLOv8Config = {
                "path": ".",
                "train": "../images",  # This is intentionally using ../ to go up to parent/images
                "names": ["person"],
            }
            config_file = dataset_dir / "config.yaml"
            with config_file.open("w") as f:
                yaml.safe_dump(config, f)

            input_obj = _YOLOv8BaseInput(input_file=config_file, input_split="train")

            assert input_obj._root_dir() == dataset_dir
            # Use .resolve() to normalize the path for comparison
            assert (
                input_obj._images_dir().resolve() == (parent_dir / "images").resolve()
            )
            assert (
                input_obj._labels_dir().resolve() == (parent_dir / "labels").resolve()
            )
