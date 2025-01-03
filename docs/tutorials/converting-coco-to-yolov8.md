# Converting COCO Labels to YOLOv8 Format

This tutorial walks you through converting object detection labels from the COCO format to the YOLOv8 format using Labelformat's CLI and Python API.

## Prerequisites

- **Labelformat Installed:** Follow the [Installation Guide](installation.md).
- **COCO Dataset:** Ensure you have a COCO-formatted dataset ready for conversion.

## Step 1: Prepare Your Dataset

Organize your dataset with the following structure:

```
project/
├── coco-labels/
│   └── train.json
├── images/
│   ├── image1.jpg
│   └── image2.jpg
```

Ensure that `train.json` contains the COCO annotations and that all images are located in the `images/` directory.

## Step 2: Using the CLI for Conversion

Open your terminal and navigate to the `project/` directory.

Run the following command to convert COCO labels to YOLOv8:

``` shell
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file coco-labels/train.json \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

### Explanation of the Command:

- `--task object-detection`: Specifies the task type.
- `--input-format coco`: Defines the input label format.
- `--input-file coco-labels/train.json`: Path to the COCO annotations file.
- `--output-format yolov8`: Desired output format.
- `--output-file yolo-labels/data.yaml`: Path to save the YOLOv8 configuration file.
- `--output-split train`: Data split label.

## Step 3: Verify the Conversion

After running the command, your project structure should include:

```
project/
├── yolo-labels/
│   ├── data.yaml
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
```

- `data.yaml`: YOLOv8 configuration file containing category names and paths.
- `labels/`: Directory containing YOLOv8-formatted label files.

**Sample `data.yaml`:**
``` yaml
names:
  0: cat
  1: dog
  2: fish
nc: 3
path: .
train: images
```

**Sample Label File (`image1.txt`):**

```
2 0.8617 0.7308 0.0359 0.0433
0 0.8180 0.6911 0.0328 0.0793
```

- **Format:** `<category_id> <center_x> <center_y> <width> <height>`
- **Coordinates:** Normalized between 0 and 1.

## Step 4: Using the Python API for Conversion

If you prefer using Python for more control, follow these steps.

### 4.1: Write the Conversion Script

Create a Python script named `coco_to_yolov8.py` with the following content:

``` python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Define input and output paths
coco_input_path = Path("coco-labels/train.json")
yolo_output_path = Path("yolo-labels/data.yaml")

# Initialize input and output classes
coco_input = COCOObjectDetectionInput(input_file=coco_input_path)
yolo_output = YOLOv8ObjectDetectionOutput(
    output_file=yolo_output_path,
    output_split="train"
)

# Perform the conversion
yolo_output.save(label_input=coco_input)

print("Conversion from COCO to YOLOv8 completed successfully!")
```

### 4.2: Execute the Script

Run the script using Python:

``` shell
python coco_to_yolov8.py
```

Upon successful execution, you will see:

```
Conversion from COCO to YOLOv8 completed successfully!
```

## Step 5: Integrate with Your Training Pipeline

Use the generated YOLOv8 labels (`data.yaml` and `labels/` directory) to train your YOLOv8 models seamlessly.

**Example YOLOv8 Training Command:**

``` shell
yolo detect train data=yolo-labels/data.yaml model=yolov8s.pt epochs=100 imgsz=640
```

## Conclusion

You've successfully converted COCO labels to YOLOv8 format using both the CLI and Python API. Labelformat simplifies label format conversions, enabling efficient integration into your computer vision projects.

---

## Next Steps

- Explore [Converting YOLOv8 to COCO](converting_yolov8_to_coco.md).
- Learn how to [Handle Labelbox Exports](handling_labelbox_exports.md).
- Dive deeper with [Advanced Usage](usage.md).

For any questions or issues, feel free to reach out via our [GitHub Issues](https://github.com/lightly-ai/labelformat/issues).