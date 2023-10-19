# Labelformat - Label Conversion, Simplified

An open-source tool to seamlessly convert between popular computer vision label formats.

**Why Labelformat:** Popular label formats are sparsely documented and store different
information. Understanding them and dealing with the differences is tedious
and time-consuming. Labelformat aims to solve this pain.

**Supported Tasks and Formats:**
- object-detection
    - [COCO](https://cocodataset.org/#format-data)
    - [KITTI](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt)
    - [Lightly](https://docs.lightly.ai/docs/prediction-format#prediction-format)
    - [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)
    - [YOLOv8](https://docs.ultralytics.com/datasets/detect/)
    - [Labelbox](https://docs.labelbox.com/reference/label-export) (input only)
- instance-segmentation
    - [COCO](https://cocodataset.org/#format-data)
    - [YOLOv8](https://docs.ultralytics.com/datasets/segment/)


> **Note**
> Labelformat is a young project, contributions and bug reports are welcome. Please see [Contributing](#contributing) section below.


## Installation

```shell
pip install labelformat
```

## Usage

### CLI

Example command:
```shell
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file coco-labels/train.json \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

#### Command Arguments

List the available tasks with:
```console
$ labelformat convert --help
usage: labelformat convert [-h] --task
                           {instance-segmentation,object-detection}

Convert labels from one format to another.

optional arguments:
  -h, --help
  --task {instance-segmentation,object-detection}
```

List the available formats for a given task with:
```console
$ labelformat convert --task object-detection --help
usage: labelformat convert [-h] --task
                           {instance-segmentation,object-detection}
                           --input-format
                           {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                           --output-format
                           {coco,kitti,labelbox,lightly,pascalvoc,yolov8}

Convert labels from one format to another.

optional arguments:
  -h, --help
  --task {instance-segmentation,object-detection}
  --input-format {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                        Input format
  --output-format {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                        Output format
```

Specify the input and output format to get required options for specific formats:
```console
$ labelformat convert \
          --task object-detection \
          --input-format coco \
          --output-format yolov8 \
          --help
usage: labelformat convert [-h] --task
                           {instance-segmentation,object-detection}
                           --input-format
                           {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                           --output-format
                           {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                           --input-file INPUT_FILE --output-file OUTPUT_FILE
                           [--output-split OUTPUT_SPLIT]

Convert labels from one format to another.

optional arguments:
  -h, --help
  --task {instance-segmentation,object-detection}
  --input-format {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                        Input format
  --output-format {coco,kitti,labelbox,lightly,pascalvoc,yolov8}
                        Output format

'coco' input arguments:
  --input-file INPUT_FILE
                        Path to input COCO JSON file

'yolov8' output arguments:
  --output-file OUTPUT_FILE
                        Output data.yaml file
  --output-split OUTPUT_SPLIT
                        Split to use
```

### Code
```python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionInput, YOLOv8ObjectDetectionOutput

label_input = COCOObjectDetectionInput(
    input_file=Path("coco-labels/train.json")
)
YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo-labels/data.yaml"),
    output_split="train",
).save(label_input=label_input)
```

## Contributing

If you encounter a bug or have a feature suggestion we will be happy if you file a GitHub issue.

We also welcome contributors, please submit a PR.

### Development

The library targets python 3.7 and higher. We use poetry to manage the development environment.

Here is an example development workflow:

```bash
# Create a virtual environment with development dependencies
poetry env use python3.7
poetry install

# Make changes
...

# Autoformat the code
poetry run make format

# Run tests
poetry run make all-checks
```

## Maintained By
[Lightly](https://www.lightly.ai) is a spin-off from ETH Zurich that helps companies 
build efficient active learning pipelines to select the most relevant data for their models.

You can find out more about the company and it's services by following the links below:

- [Homepage](https://www.lightly.ai)
- [Web-App](https://app.lightly.ai)
- [Lightly Solution Documentation (Lightly Worker & API)](https://docs.lightly.ai/)
