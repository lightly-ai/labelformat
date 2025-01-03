
# Quick Start Guide

Get up and running with **Labelformat** in minutes! This Quick Start Guide provides simple, copy-paste examples to help you convert label formats effortlessly.

## Scenario 1: Convert COCO to YOLOv8 Using CLI

### Step 1: Prepare Your Files

Ensure you have the following structure:
```
project/
├── coco-labels/
│   └── train.json
├── images/
│   ├── image1.jpg
│   └── image2.jpg
```

### Step 2: Run the Conversion Command

Open your terminal, navigate to your project directory, and execute:

```shell
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file coco-labels/train.json \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

### Step 3: Verify the Output

Your project structure should now include:

```
project/
├── yolo-labels/
│   ├── data.yaml
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
```

---

## Scenario 2: Convert YOLOv8 to COCO Using Python API

### Step 1: Install Labelformat

If you haven't installed Labelformat yet, do so via pip:
``` shell
pip install labelformat
```

### Step 2: Write the Conversion Script

Create a Python script, `convert_yolo_to_coco.py`, with the following content:

``` python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionOutput, YOLOv8ObjectDetectionInput

# Load YOLOv8 labels
yolo_input = YOLOv8ObjectDetectionInput(
    input_file=Path("yolo-labels/data.yaml"),
    input_split="train"
)

# Convert to COCO format and save
coco_output = COCOObjectDetectionOutput(
    output_file=Path("coco-from-yolo/converted_coco.json")
)
coco_output.save(label_input=yolo_input)

print("Conversion from YOLOv8 to COCO completed successfully!")
```

### Step 3: Execute the Script

Run the script:

``` shell
python convert_yolo_to_coco.py
```

### Step 4: Check the COCO Output

Your project should now have:

```
project/
├── coco-from-yolo/
│   └── converted_coco.json
```

---

## Scenario 3: Convert Labelbox Export to Lightly Format

### Step 1: Export Labels from Labelbox

Ensure you have the Labelbox export file, e.g., `labelbox-export.ndjson`.

### Step 2: Run the Conversion Command

``` shell
labelformat convert \
    --task object-detection \
    --input-format labelbox \
    --input-file labelbox-export.ndjson \
    --category-names cat,dog,fish \
    --output-format lightly \
    --output-folder lightly-labels/annotation-task
```

### Step 3: Verify the Lightly Output

Your project structure should include:

```
project/
├── lightly-labels/
│   ├── annotation-task/
│   │   ├── schema.json
│   │   ├── image1.json
│   │   └── image2.json
```