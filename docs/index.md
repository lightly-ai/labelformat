
![Labelformat Banner](assets/labelformat_banner.png)

# Labelformat - Fast Label Conversion for Computer Vision

**Labelformat** is an open-source Python framework for converting between popular computer vision annotation formats like YOLO, COCO, PascalVOC, and KITTI. Save hours on tedious format conversions and ensure consistency in your workflows.

!!! tip
    Check out our [LightlyStudio](https://github.com/lightly-ai/lightly-studio) open source project that builds on top of Labelformat to visualize and edit your annotation labels.

## Key Features
- **Wide Format Support**: COCO, YOLO (v5-v12, v26), PascalVOC, KITTI, Labelbox, RT-DETR, RT-DETRv2, and more.
- **Cross-Platform**: Compatible with Python 3.8 through 3.14 on Windows, macOS, and Linux.
- **Flexible Usage**: Intuitive CLI and Python API.
- **Efficient**: Memory-conscious, optimized for large datasets.
- **Offline First**: Operates locally without data uploads.
- **Tested for Accuracy**: Round-trip tests for consistent results.

## Get Started Quickly

1. **Install via pip**:
    ```bash
    pip install labelformat
    ```
2. **Convert Labels in One Command**:
    ```bash
    labelformat convert --task object-detection \
                        --input-format coco \
                        --input-file coco-labels/train.json \
                        --output-format yolov8 \
                        --output-file yolo-labels/data.yaml
    ```

## Supported Formats

### **2D Object Detection Label Formats**

| Format       | Read ✔️ | Write ✔️ |
|--------------|---------|----------|
| COCO         | ✔️      | ✔️       |
| KITTI        | ✔️      | ✔️       |
| Labelbox     | ✔️      | ❌       |
| Lightly      | ✔️      | ✔️       |
| PascalVOC    | ✔️      | ✔️       |
| RT-DETR      | ✔️      | ✔️       |
| RT-DETRv2    | ✔️      | ✔️       |
| YOLOv5 - v12, v26 | ✔️      | ✔️       |

---

### **2D Instance Segmentation Label Formats**

| Format       | Read ✔️ | Write ✔️ |
|--------------|---------|----------|
| COCO         | ✔️      | ✔️       |
| YOLOv8       | ✔️      | ✔️       |


## Explore More
- [Quick Start Guide](quick-start.md)
- [Detailed Usage Guide](usage.md)
- [List of all features](features.md)
---

## 📦 Quick Links

- [GitHub Repository](https://github.com/lightly-ai/labelformat)
- [PyPI Package](https://pypi.org/project/labelformat/)
- [Documentation](https://labelformat.com)

Labelformat is maintained by [Lightly](https://www.lightly.ai).
