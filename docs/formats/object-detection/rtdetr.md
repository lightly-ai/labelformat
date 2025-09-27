# RT-DETR Object Detection Format

## Overview

**RT-DETR (Real-Time DEtection TRansformer)** is a groundbreaking end-to-end object detection framework introduced in the paper [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069). RT-DETR represents the first real-time end-to-end object detector that successfully challenges the dominance of YOLO detectors in real-time applications. Unlike traditional detectors that require Non-Maximum Suppression (NMS) post-processing, RT-DETR eliminates NMS entirely while achieving superior speed and accuracy performance.

> **Info:** RT-DETR was introduced through the academic paper "DETRs Beat YOLOs on Real-time Object Detection" published in 2023.
  For the full paper, see: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
  For implementation details and code, see: [GitHub Repository: lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

> **Availability:** RT-DETR is now available in multiple frameworks:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/rt_detr)
  - [Ultralytics](https://docs.ultralytics.com/models/rtdetr/)

## Key RT-DETR Model Features

RT-DETR uses the standard **COCO annotation format** while introducing revolutionary architectural innovations for real-time detection:

- **End-to-End Architecture:** First real-time detector to completely eliminate NMS post-processing, providing more stable and predictable inference times.
- **Efficient Hybrid Encoder:** Novel encoder design that decouples intra-scale interaction and cross-scale fusion to significantly reduce computational overhead.
- **Uncertainty-Minimal Query Selection:** Advanced query initialization scheme that optimizes both classification and localization confidence for improved detection quality.
- **Flexible Speed Tuning:** Supports adjustable inference speed by modifying the number of decoder layers without retraining.
- **Superior Performance:** Achieves state-of-the-art results (e.g., RT-DETR-R50 reaches 53.1% mAP @ 108 FPS on T4 GPU, outperforming YOLOv8-L in both speed and accuracy).
- **Multiple Model Scales:** Available in various scales (R18, R34, R50, R101) to accommodate different computational requirements.

These architectural innovations are handled internally by the model design and training pipeline, requiring no changes to the standard COCO annotation format described below.

## Specification of RT-DETR Detection Format

RT-DETR uses the standard **COCO format** for annotations, ensuring seamless integration with existing COCO datasets and tools. The format consists of a single JSON file containing three main components:

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

## Directory Structure of RT-DETR Dataset

```
dataset/
├── images/                   # Image files
│   ├── image1.jpg
│   └── image2.jpg
└── annotations.json         # Single JSON file containing all annotations
```

## Benefits of RT-DETR Format

- **Standard Compatibility:** Uses the widely-adopted COCO format, ensuring compatibility with existing tools and frameworks.
- **End-to-End Processing:** Eliminates NMS post-processing, providing more stable and predictable inference performance.
- **Flexibility:** Supports adjustable inference speeds without retraining, making it adaptable to various real-time scenarios.
- **Superior Accuracy:** Achieves better accuracy than comparable YOLO detectors while maintaining competitive speed.

## Converting Annotations to RT-DETR Format with Labelformat

Since RT-DETR uses the standard COCO format, converting annotations to RT-DETR format is equivalent to converting to COCO format.

### Installation

First, ensure that Labelformat is installed:

```shell
pip install labelformat
```

### Conversion Example: YOLOv8 to RT-DETR

Assume you have annotations in YOLOv8 format and wish to convert them to RT-DETR. Here's how you can achieve this using Labelformat.

**Step 1: Prepare Your Dataset**

Ensure your dataset follows the standard YOLOv8 structure with `data.yaml` and label files.

**Step 2: Run the Conversion Command**

Use the Labelformat CLI to convert YOLOv8 annotations to RT-DETR (COCO format):
```bash
labelformat convert \
    --task object-detection \
    --input-format yolov8 \
    --input-file dataset/data.yaml \
    --input-split train \
    --output-format rtdetr \
    --output-file dataset/rtdetr_annotations.json
```

**Step 3: Verify the Converted Annotations**

After conversion, your dataset structure will be:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── rtdetr_annotations.json    # COCO format annotations for RT-DETR
```

### Python API Example

```python
from pathlib import Path
from labelformat.formats import YOLOv8ObjectDetectionInput, RTDETRObjectDetectionOutput

# Load YOLOv8 format
label_input = YOLOv8ObjectDetectionInput(
    input_file=Path("dataset/data.yaml"),
    input_split="train"
)

# Convert to RT-DETR format
RTDETRObjectDetectionOutput(
    output_file=Path("dataset/rtdetr_annotations.json")
).save(label_input=label_input)
```

## Performance Benchmarks

RT-DETR achieves impressive performance across different model scales:

| Model | Input | Dataset | mAP (%) | mAP50 (%) | Parameters (M) | GFLOPs | FPS (T4) |
|-------|-------|---------|---------|-----------|----------------|--------|----------|
| RT-DETR-R18 | 640 | COCO | 46.5 | 63.8 | 20 | 60 | 217 |
| RT-DETR-R34 | 640 | COCO | 48.9 | 66.8 | 31 | 92 | 161 |
| RT-DETR-R50-m | 640 | COCO | 51.3 | 69.6 | 36 | 100 | 145 |
| RT-DETR-R50 | 640 | COCO | 53.1 | 71.3 | 42 | 136 | 108 |
| RT-DETR-R101 | 640 | COCO | 54.3 | 72.7 | 76 | 259 | 74 |
| **RT-DETR-R18** | 640 | **COCO + Objects365** | **49.2** | **66.6** | 20 | 60 | **217** |
| **RT-DETR-R50** | 640 | **COCO + Objects365** | **55.3** | **73.4** | 42 | 136 | **108** |
| **RT-DETR-R101** | 640 | **COCO + Objects365** | **56.2** | **74.6** | 76 | 259 | **74** |

*Performance measured on T4 GPU with TensorRT FP16 precision*

## Error Handling in Labelformat

Since RT-DETR uses the COCO format, the same validation and error handling applies:

- **Invalid JSON Structure:** Proper error reporting for malformed JSON files
- **Missing Required Fields:** Validation ensures all required COCO fields are present
- **Reference Integrity:** Checks that image_id and category_id references are valid
- **Bounding Box Validation:** Ensures bounding boxes are within image boundaries

Example of a properly formatted annotation:
```json
{
  "images": [{"id": 0, "file_name": "image1.jpg", "width": 640, "height": 480}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [{"image_id": 0, "category_id": 1, "bbox": [100, 120, 50, 80]}]
}
```