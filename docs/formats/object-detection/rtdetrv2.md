# RT-DETRv2 Object Detection Format

## Overview

**RT-DETRv2** is an enhanced version of the Real-Time DEtection TRansformer ([RT-DETR](https://arxiv.org/abs/2304.08069)), introduced in the paper [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140). Building upon the groundbreaking end-to-end object detection framework of the original RT-DETR, RT-DETRv2 continues the legacy of eliminating Non-Maximum Suppression (NMS) post-processing while introducing additional improvements in accuracy and efficiency for real-time object detection scenarios.

> **Info:** RT-DETRv2 was introduced through the technical report "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer" published in 2024.
  For the full paper, see: [arXiv:2407.17140](https://arxiv.org/abs/2407.17140)
  For RT-DETR foundation, see: [RT-DETR Paper (arXiv:2304.08069)](https://arxiv.org/abs/2304.08069)
  For implementation details and code, see: [GitHub Repository: lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

> **Availability:** RT-DETRv2 is now available in multiple frameworks:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)
  - [Ultralytics](https://docs.ultralytics.com/models/rtdetr/)

## Key RT-DETRv2 Model Features

RT-DETRv2 maintains compatibility with the standard **COCO annotation format** while introducing specific technical improvements over RT-DETR:

- **Distinct Sampling Points for Different Scales:** Introduces flexible multi-scale feature extraction by setting different numbers of sampling points for features at different scales in the deformable attention module, rather than using the same number across all scales.
- **Discrete Sampling Operator:** Provides an optional discrete sampling operator to replace the grid_sample operator, removing deployment constraints typically associated with DETRs and improving practical applicability across different deployment platforms.
- **Dynamic Data Augmentation:** Implements adaptive data augmentation strategy that applies stronger augmentation in early training periods and reduces it in later stages to improve model robustness and target domain adaptation.
- **Scale-Adaptive Hyperparameters:** Customizes optimizer hyperparameters based on model scale, using higher learning rates for lighter models (e.g., ResNet18) and lower rates for larger models (e.g., ResNet101) to achieve optimal performance.
- **Bag-of-Freebies Approach:** Incorporates multiple training improvements that enhance performance without increasing inference cost or model complexity.
- **Consistent Performance Gains:** Achieves improved accuracy across all model scales (S: +1.4 mAP, M: +1.0 mAP, L: +0.3 mAP) while maintaining the same inference speed as RT-DETR.

These enhancements are handled internally by the model design and training pipeline, requiring no changes to the standard COCO annotation format described below.

## Specification of RT-DETRv2 Detection Format

RT-DETRv2 uses the standard **COCO format** for annotations, ensuring complete compatibility with existing COCO datasets and tools. The format specification is identical to the original COCO format:

### `images`
Defines metadata for each image in the dataset:
```json
{
  "id": 0,                    // Unique image ID
  "file_name": "image1.jpg",  // Image filename
  "width": 640,               // Image width in pixels
  "height": 416               // Image height in pixels
}
```

### `categories`
Defines the object classes:
```json
{
  "id": 0,                    // Unique category ID
  "name": "cat"               // Category name
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

## Directory Structure of RT-DETRv2 Dataset

```
dataset/
├── images/                   # Image files
│   ├── image1.jpg
│   └── image2.jpg
└── annotations.json         # Single JSON file containing all annotations
```

## Benefits of RT-DETRv2 Format

- **Standard Compatibility:** Uses the widely-adopted COCO format, ensuring compatibility with existing tools and frameworks.
- **End-to-End Processing:** Maintains the NMS-free architecture for stable and predictable inference performance.
- **Enhanced Performance:** Improved accuracy and efficiency compared to the original RT-DETR.

## Converting Annotations to RT-DETRv2 Format with Labelformat

Since RT-DETRv2 uses the standard COCO format, converting annotations to RT-DETRv2 format is equivalent to converting to COCO format.

### Installation

First, ensure that Labelformat is installed:

```shell
pip install labelformat
```

### Conversion Example: YOLOv8 to RT-DETRv2

**Step 1: Prepare Your Dataset**

Ensure your dataset follows the standard YOLOv8 structure with `data.yaml` and label files.

**Step 2: Run the Conversion Command**

Use the Labelformat CLI to convert YOLOv8 annotations to RT-DETRv2 (COCO format):
```bash
labelformat convert \
    --task object-detection \
    --input-format yolov8 \
    --input-file dataset/data.yaml \
    --input-split train \
    --output-format rtdetrv2 \
    --output-file dataset/rtdetrv2_annotations.json
```

**Step 3: Verify the Converted Annotations**

After conversion, your dataset structure will be:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── rtdetrv2_annotations.json    # COCO format annotations for RT-DETRv2
```

### Python API Example

```python
from pathlib import Path
from labelformat.formats import YOLOv8ObjectDetectionInput, RTDETRv2ObjectDetectionOutput

# Load YOLOv8 format
label_input = YOLOv8ObjectDetectionInput(
    input_file=Path("dataset/data.yaml"),
    input_split="train"
)

# Convert to RT-DETRv2 format
RTDETRv2ObjectDetectionOutput(
    output_file=Path("dataset/rtdetrv2_annotations.json")
).save(label_input=label_input)
```

## RT-DETRv2 vs RT-DETR

RT-DETRv2 builds upon the foundation of RT-DETR with several key improvements:

- **Enhanced Architecture:** Refined encoder and decoder designs for better performance
- **Improved Training:** Advanced training strategies and optimization techniques
- **Better Accuracy:** Higher detection accuracy across various model scales

## Error Handling in Labelformat

Since RT-DETRv2 uses the COCO format, the same validation and error handling applies:

- **Invalid JSON Structure:** Proper error reporting for malformed JSON files
- **Missing Required Fields:** Validation ensures all required COCO fields are present
- **Invalid JSON Structure:** Proper error reporting for malformed JSON files.
- **Missing Required Fields:** Validation ensures all required COCO fields are present.
- **Reference Integrity:** Checks that image_id and category_id references are valid.
- **Bounding Box Validation:** Ensures bounding boxes are within image boundaries.
```json
{
  "images": [{"id": 0, "file_name": "image1.jpg", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [{"image_id": 0, "category_id": 1, "bbox": [100, 120, 50, 80]}]
}
```