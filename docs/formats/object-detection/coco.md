# COCO Object Detection Format

## Overview
COCO (Common Objects in Context) is a large-scale object detection dataset format developed by Microsoft. The format has become one of the most widely adopted standards for object detection tasks. You can find the complete format specification in the [official COCO documentation](https://cocodataset.org/#format-data).

## Specification of COCO Detection Format

COCO uses a single JSON file containing all annotations. The format consists of three main components:

- **Images:** Defines metadata for each image in the dataset.
- **Categories:** Defines the object classes.
- **Annotations:** Defines object instances.

### Images
Defines metadata for each image in the dataset:
```json
{
  "id": 0,                    // Unique image ID
  "file_name": "image1.jpg",  // Image filename
  "width": 640,              // Image width in pixels
  "height": 416              // Image height in pixels
}
```

### Categories
Defines the object classes:
```json
{
  "id": 0,                    // Unique category ID
  "name": "cat"              // Category name
}
```

### Annotations
Defines object instances:
```json
{
  "image_id": 0,              // Reference to image
  "category_id": 2,           // Reference to category
  "bbox": [540.0, 295.0, 23.0, 18.0]  // [x, y, width, height] in absolute pixels
}
```

## Directory Structure of COCO Dataset
```
dataset/
├── images/                   # Image files
│   ├── image1.jpg
│   └── image2.jpg
└── annotations.json         # Single JSON file containing all annotations
```

## Converting with Labelformat

### Command Line Interface
Convert COCO format to YOLOv8:
```bash
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file coco-labels/annotations.json \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

Convert YOLOv8 format to COCO:
```bash
labelformat convert \
    --task object-detection \
    --input-format yolov8 \
    --input-file yolo-labels/data.yaml \
    --input-split train \
    --output-format coco \
    --output-file coco-labels/annotations.json
```

### Python API
```python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Load COCO format
label_input = COCOObjectDetectionInput(
    input_file=Path("coco-labels/annotations.json")
)

# Convert to YOLOv8 format
YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo-labels/data.yaml"),
    output_split="train",
).save(label_input=label_input)
```

## Example
Complete annotations.json example:
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 416
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "cat"
    }
  ],
  "annotations": [
    {
      "image_id": 0,
      "category_id": 0,
      "bbox": [540.0, 295.0, 23.0, 18.0]
    }
  ]
}
``` 