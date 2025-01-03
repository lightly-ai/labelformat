# usage.md

# Detailed Usage Guide

Labelformat offers both a Command-Line Interface (CLI) and a Python API to cater to different workflows. This guide provides in-depth instructions on how to use both interfaces effectively.

## Table of Contents

- [CLI Usage](#cli-usage)
  - [Basic Conversion Command](#basic-conversion-command)
  - [Advanced CLI Options](#advanced-cli-options)
- [Python API Usage](#python-api-usage)
  - [Basic Conversion](#basic-conversion)
  - [Customizing Conversion](#customizing-conversion)
- [Common Tasks](#common-tasks)
  - [Handling Category Names](#handling-category-names)
  - [Managing Image Paths](#managing-image-paths)

---

## CLI Usage

Labelformat's CLI provides a straightforward way to convert label formats directly from the terminal.

### Basic Conversion Command

**Example:** Convert Object Detection labels from COCO to YOLOv8.

``` shell
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file path/to/coco/train.json \
    --output-format yolov8 \
    --output-file path/to/yolo/data.yaml \
    --output-split train
```

**Parameters:**

- `--task`: Specify the task type (`object-detection` or `instance-segmentation`).
- `--input-format`: The format of the input labels (e.g., `coco`).
- `--input-file` or `--input-folder`: Path to the input label file or folder.
- `--output-format`: The desired output label format (e.g., `yolov8`).
- `--output-file` or `--output-folder`: Path to save the converted labels.
- `--output-split`: Define the data split (`train`, `val`, `test`).

### Advanced CLI Options

**Listing Supported Formats:**

To see all supported input and output formats for a specific task:

``` shell
labelformat convert --task object-detection --help
```

**Specifying Category Names:**

Some formats require explicit category names. Use the `--category-names` argument:

``` shell
labelformat convert \
    --task object-detection \
    --input-format labelbox \
    --input-file labelbox-export.ndjson \
    --category-names cat,dog,fish \
    --output-format coco \
    --output-file coco-output/train.json \
    --output-split train
```

**Handling Missing Images:**

When converting formats that require image files (e.g., YOLO to COCO), ensure your image paths are correctly specified. Use `--images-rel-path` to define the relative path to images:

``` shell
labelformat convert \
    --task object-detection \
    --input-format kitti \
    --input-folder kitti-labels/labels \
    --images-rel-path ../images \
    --output-format pascalvoc \
    --output-folder pascalvoc-labels
```

## Python API Usage

For more flexible integrations, Labelformat provides a Python API.

### Basic Conversion

**Example:** Convert COCO to YOLOv8.

``` python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Initialize input and output classes
coco_input = COCOObjectDetectionInput(input_file=Path("coco-labels/train.json"))
yolo_output = YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo-labels/data.yaml"),
    output_split="train"
)

# Perform the conversion
yolo_output.save(label_input=coco_input)

print("Conversion from COCO to YOLOv8 completed successfully!")
```

### Customizing Conversion

**Example:** Adding Custom Fields or Handling Special Cases.

``` python
from pathlib import Path
from labelformat.formats import COCOInstanceSegmentationInput, YOLOv8InstanceSegmentationOutput

# Initialize input for instance segmentation
coco_inst_input = COCOInstanceSegmentationInput(input_file=Path("coco-instance/train.json"))

# Initialize YOLOv8 instance segmentation output
yolo_inst_output = YOLOv8InstanceSegmentationOutput(
    output_file=Path("yolo-instance-labels/data.yaml"),
    output_split="train"
)

# Perform the conversion
yolo_inst_output.save(label_input=coco_inst_input)

print("Instance segmentation conversion completed successfully!")
```

## Common Tasks

### Handling Category Names

Some label formats require you to specify category names explicitly. Ensure that category names are consistent across your dataset.

**Example:**

``` shell
labelformat convert \
    --task object-detection \
    --input-format labelbox \
    --input-file labelbox-export.ndjson \
    --category-names cat,dog,fish \
    --output-format coco \
    --output-file coco-output/train.json \
    --output-split train
```

### Managing Image Paths

When converting formats that reference image files, accurately specify the relative paths to avoid missing files.

**Example:**

``` shell
labelformat convert \
    --task object-detection \
    --input-format kitti \
    --input-folder kitti-labels/labels \
    --images-rel-path ../images \
    --output-format pascalvoc \
    --output-folder pascalvoc-labels
```

---

## Tips and Best Practices

- **Backup Your Data:** Always keep a backup of your original labels before performing conversions.
- **Validate Output:** After conversion, verify the output labels to ensure accuracy.
- **Consistent Naming:** Maintain consistent naming conventions for categories and files across different formats.
- **Leverage Round-Trip Tests:** Use Labelformat's testing capabilities to ensure label consistency when converting back and forth between formats.

For more detailed examples and advanced usage scenarios, explore our [Tutorials](tutorials/converting_coco_to_yolov8.md) section.