# YOLOv5 Object Detection Format

## Overview

**YOLOv5** is a well-known object detection model in the **You Only Look Once (YOLO)** series, renowned for its real-time object detection capabilities. Building upon the foundations of YOLO series, YOLOv5 introduces significant advancements in model architecture and training methodologies, enhancing both accuracy and efficiency. Despite these improvements, YOLOv5 retains the same **object detection format** as its predecessors, utilizing normalized coordinates in text files. This consistency ensures seamless integration and compatibility with existing workflows while benefiting from YOLOv5's enhanced performance.

> **Info:** Unlike other YOLO versions, YOLOv5 was not introduced through an academic paper. The most commonly referenced implementation is the Ultralytics YOLOv5 repository on GitHub, which has become the de facto standard implementation. 
  For implementation details and specifications, see: [GitHub Repository: ultralytics/yolov5](https://github.com/ultralytics/yolov5)


## Specification of YOLOv5 Detection Format

The **YOLOv5 detection format** remains consistent with previous versions, ensuring ease of adoption and compatibility. Below are the detailed specifications:

- **One Text File per Image:**  
  For every image in your dataset, there exists a corresponding `.txt` file containing annotation data.

- **Object Representation:**  
  Each line in the text file represents a single object detected within the image, following the format: `<class_id> <x_center> <y_center> <width> <height>`  
    - **`<class_id>` (Integer):**   An integer representing the object's class.
    - **`<x_center>` and `<y_center>` (Float):** The normalized coordinates of the object's center relative to the image's width and    height.
    - **`<width>` and `<height>` (Float):** The normalized width and height of the bounding box encompassing the object.

- **Normalization of Values:**  
  All coordinate and size values are normalized to a range between `0.0` and `1.0`.  
  - **Normalization Formula:**  
    To convert pixel values to normalized coordinates:  
    ```markdown
    normalized_x = x_pixel / image_width  
    normalized_y = y_pixel / image_height  
    normalized_width = box_width_pixel / image_width  
    normalized_height = box_height_pixel / image_height
    ```

- **Class ID Indexing:**  
  Class IDs start from `0`, with each ID corresponding to a specific object category defined in the `data.yaml` configuration file. Class IDs must be contiguous integers (e.g., 0,1,2 and not 0,2,3) to ensure proper model training and inference.

- **Configuration via `data.yaml`:**  
  The `data.yaml` file contains essential configuration settings, including paths to training and validation datasets, number of classes (`nc`), and a mapping of class names to their respective IDs (`names`).

## Directory Structure Requirements

The dataset must maintain a parallel directory structure for images and their corresponding label files. There are two common organizational patterns:

### Pattern 1: Images and Labels as Root Directories
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

### Pattern 2: Train/Val as Root Directories
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── val/
    ├── images/
    │   ├── image3.jpg
    │   └── image4.jpg
    └── labels/
        ├── image3.txt
        └── image4.txt
```

The corresponding `data.yaml` configuration should match your chosen structure:

```yaml
# For Pattern 1:
path: .  # Optional - defaults to current directory if omitted
train: images/train  # Path to training images
val: images/val      # Path to validation images

# For Pattern 2:
path: .  # Optional - defaults to current directory if omitted
train: train/images  # Path to training images
val: val/images      # Path to validation images
```

**Important:** Label files must have the same name as their corresponding image files (excluding the file extension) and must maintain the parallel directory structure, only replacing `images` with `labels` in the path. 

## Benefits of YOLOv5 Format

- Simplicity: Easy to read and write, facilitating quick dataset preparation.
- Efficiency: Compact representation reduces storage requirements.
- Compatibility: Maintains consistency across YOLO versions, ensuring seamless integration with various tools and frameworks.

## Example of YOLOv5 Format

**Example of `data.yaml`:**
```yaml
path: .  # Dataset root directory (defaults to current directory if omitted). Can also be `../dataset`.
train: images/train  # Directory for training images
val: images/val  # Directory for validation images
test: images/test  # Directory for test images (optional)

names:
  0: cat
  1: dog
  2: person
```

**Example Annotation**

For an image named `image1.jpg`, the corresponding `image1.txt` might contain:
```
0 0.716797 0.395833 0.216406 0.147222
1 0.687500 0.379167 0.255208 0.175000
```

**Explanation:**
- The first line represents an object of class `0` (e.g., `cat`) with its bounding box centered at `(0.716797, 0.395833)` relative to the image dimensions, and a width and height of `0.216406` and `0.147222` respectively.
- The second line represents an object of class `1` (e.g., `dog`) with its own bounding box specifications.

### Normalizing Bounding Box Coordinates for YOLOv5

To convert pixel values to normalized values required by YOLOv5:

```python
# Given pixel values and image dimensions
x_top_left = 150     # x coordinate of top-left corner of bounding box
y_top_left = 200     # y coordinate of top-left corner of bounding box
width_pixel = 50     # width of bounding box
height_pixel = 80    # height of bounding box
image_width = 640
image_height = 480

# 1. Convert top-left coordinates to center coordinates (in pixels)
x_center_pixel = x_top_left + (width_pixel / 2)   # 150 + (50/2) = 175
y_center_pixel = y_top_left + (height_pixel / 2)  # 200 + (80/2) = 240

# 2. Normalize all values (divide by image dimensions)
x_center = x_center_pixel / image_width     # 175 / 640 = 0.273438
y_center = y_center_pixel / image_height    # 240 / 480 = 0.500000
width = width_pixel / image_width           # 50 / 640 = 0.078125
height = height_pixel / image_height        # 80 / 480 = 0.166667

# Annotation line format: <class_id> <x_center> <y_center> <width> <height>
annotation = f"0 {x_center} {y_center} {width} {height}"
# Output: "0 0.273438 0.500000 0.078125 0.166667"
```

## Converting Annotations to YOLOv5 Format with Labelformat

Our **Labelformat** framework simplifies the process of converting various annotation formats to the YOLOv5 detection format. Below is a step-by-step guide to perform this conversion.

### Installation

First, ensure that Labelformat is installed. You can install it via pip:

```shell
pip install labelformat
```

### Conversion Example: COCO to YOLOv5

Assume you have annotations in the COCO format and wish to convert them to YOLOv5. Here’s how you can achieve this using Labelformat.

**Step 1: Prepare Your Dataset**

Ensure your dataset follows the standard COCO structure:

- You have a `.json` file with the COCO annotations. (e.g. `annotations/instances_train.json`)
- You have a directory with the images. (e.g. `images/`)

Full example:
```bash
dataset/
├── annotations/
│   └── instances_train.json
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

**Step 2: Run the Conversion Command**

Use the Labelformat CLI to convert COCO annotations to YOLOv5:
```
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file dataset/annotations/instances_train.json \
    --output-format yolov5 \
    --output-folder dataset/yolov5_labels \
    --output-split train
```

**Step 3: Verify the Converted Annotations**

After conversion, your dataset structure will be:
```
dataset/
├── yolov5_labels/
│   ├── data.yaml
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
```

Contents of `data.yaml`:
```yaml
path: .  # Dataset root directory
train: images  # Directory for training images
nc: 3  # Number of classes
names:  # Class name mapping
  0: cat
  1: dog
  2: person
```

Contents of `image1.txt`:
```
0 0.234375 0.416667 0.078125 0.166667
1 0.500000 0.500000 0.100000 0.200000
```

## Error Handling in Labelformat

The format implementation includes several safeguards:

- **Missing Label Files:**
  - Warning is logged if a label file doesn't exist for an image
  - Image is skipped from processing
  
- **File Access Issues:**
  - Errors are logged if label files cannot be read due to permissions or other OS issues
  - Affected images are skipped from processing

- **Label Format Validation:**
  - Each line must contain exactly 5 space-separated values
  - Invalid lines are logged as warnings and skipped
  - All values must be convertible to appropriate types:
    - Category ID must be a valid integer
    - Coordinates and dimensions must be valid floats
  - Category IDs must exist in the category mapping

Example of a properly formatted label file:
```text
0 0.716797 0.395833 0.216406 0.147222  # Each line must have exactly 5 space-separated values
1 0.687500 0.379167 0.255208 0.175000  # All values must be valid numbers within [0,1] range
``` 