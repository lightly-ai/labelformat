# YOLOv26 Object Detection Format

## Overview

**YOLOv26** is the latest evolution in the **You Only Look Once (YOLO)** series, engineered specifically for edge and low-power devices. It introduces a streamlined design that removes unnecessary complexity while integrating targeted innovations to deliver faster, lighter, and more accessible deployment. YOLOv26 uses the **same object detection format** as YOLOv8-v12, utilizing normalized coordinates in text files for seamless compatibility.

> **Info:** YOLOv26 is currently in preview and under development. Performance numbers are preliminary and final releases will follow soon. For the latest updates, see: [GitHub Repository: ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## Key YOLOv26 Features

YOLOv26 maintains full compatibility with the YOLOv8-v12 label format while introducing several breakthrough innovations:

- **End-to-End NMS-Free Inference:** Native end-to-end model producing predictions directly without non-maximum suppression, reducing latency and simplifying deployment
- **DFL Removal:** Eliminates Distribution Focal Loss module for better export compatibility and broader hardware support on edge devices
- **MuSGD Optimizer:** Hybrid optimizer combining SGD with Muon, inspired by Moonshot AI's Kimi K2 breakthroughs in LLM training
- **ProgLoss + STAL:** Enhanced loss functions with notable improvements in small-object detection accuracy
- **43% Faster CPU Inference:** Specifically optimized for edge computing with significant CPU performance gains

## Format Specification

YOLOv26 uses the **identical format** as YOLOv8, YOLOv9, YOLOv10, and YOLOv11. Please refer to the [YOLOv8 format documentation](yolov8.md) for complete format specifications, including:

- Text file structure with normalized coordinates
- Directory organization patterns
- Configuration via `data.yaml`
- Coordinate normalization formulas
- Example annotations

## Converting Annotations to YOLOv26 Format

Since YOLOv26 uses the same format as YOLOv8-v11, you can convert from other formats using Labelformat:

```bash
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file dataset/annotations/instances_train.json \
    --output-format yolov26 \
    --output-folder dataset/yolov26_labels \
    --output-split train
```

The converted output will be fully compatible with YOLOv26 training and inference pipelines.

## Supported Tasks

YOLOv26 extends YOLO's versatility across multiple computer vision tasks:

| Model | Task | Inference | Validation | Training | Export |
|-------|------|-----------|------------|----------|---------|
| YOLO26 | Detection | ✅ | ✅ | ✅ | ✅ |
| YOLO26-seg | Instance Segmentation | ✅ | ✅ | ✅ | ✅ |
| YOLO26-pose | Pose/Keypoints | ✅ | ✅ | ✅ | ✅ |
| YOLO26-obb | Oriented Detection | ✅ | ✅ | ✅ | ✅ |
| YOLO26-cls | Classification | ✅ | ✅ | ✅ | ✅ |