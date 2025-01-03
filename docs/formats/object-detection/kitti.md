# KITTI Object Detection Format

## Overview
The KITTI format was developed as part of the KITTI Vision Benchmark Suite, focusing on autonomous driving scenarios. This format is particularly well-suited for 3D object detection and tracking tasks. The complete format specification can be found in the [KITTI development kit documentation](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt).

## Specification of KITTI Detection Format
Each object is represented by 15 space-separated values:

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Object type (Car, Van, Truck, etc.)
   1    truncated    Float 0-1 (truncated ratio)
   1    occluded     Integer (0=visible, 1=partly occluded, 2=fully occluded)
   1    alpha        Observation angle (-pi..pi)
   4    bbox         2D bounding box (x1,y1,x2,y2) in pixels
   3    dimensions   3D dimensions (height, width, length) in meters
   3    location     3D location (x,y,z) in camera coordinates
   1    rotation_y   Rotation around Y-axis in camera coordinates
```

## Directory Structure of KITTI Dataset
```
dataset/
├── images/
│   ├── 000000.png
│   └── 000001.png
└── labels/
    ├── 000000.txt
    └── 000001.txt
```

## Label Format
```
# Example: 000000.txt
Car -1 -1 -10 614 181 727 284 -1 -1 -1 -1000 -1000 -1000 -10
Pedestrian -1 -1 -10 123 456 789 012 -1 -1 -1 -1000 -1000 -1000 -10
```

Note: The filename of each label file must match its corresponding image file, with .txt extension.

## Annotation Format Conversion
### Using CLI
Convert from YOLOv8 to KITTI format:
```bash
labelformat convert \
    --task object-detection \
    --input-format yolov8 \
    --input-file yolo-labels/data.yaml \
    --input-split train \
    --output-format kitti \
    --output-folder kitti-labels
```

Convert from KITTI to YOLOv8 format:
```bash
labelformat convert \
    --task object-detection \
    --input-format kitti \
    --input-folder kitti-labels \
    --category-names car,pedestrian,cyclist \
    --images-rel-path ../images \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

### Using Python
```python
from pathlib import Path
from labelformat.formats import KittiObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Load KITTI labels
label_input = KittiObjectDetectionInput(
    input_folder=Path("kitti-labels"),
    category_names="car,pedestrian,cyclist",
    images_rel_path="../images"
)

# Convert to YOLOv8 and save
YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo-labels/data.yaml"),
    output_split="train"
).save(label_input=label_input)
```

## Notes
- KITTI format uses absolute pixel coordinates (x1,y1,x2,y2) for bounding boxes
- Some fields like truncated, occluded, dimensions etc. are optional and can be set to -1 if unknown
- The category name (type) should match one of the predefined categories when converting 