![Labelformat - Label Conversion, Simplified](labelformat_banner.png?raw=true "Labelformat")

# Labelformat - Label Conversion, Simplified

![GitHub](https://img.shields.io/github/license/lightly-ai/labelformat)
![Unit Tests](https://github.com/lightly-ai/labelformat/workflows/Run%20Tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/labelformat)](https://pypi.org/project/labelformat/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An open-source tool to seamlessly convert between popular computer vision label formats.

#### Why Labelformat

Popular label formats are sparsely documented and store different
information. Understanding them and dealing with the differences is tedious
and time-consuming. Labelformat aims to solve this pain.

#### Supported Tasks and Formats

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

#### Features

- Support for common dataset label formats (more coming soon)
- Support for common tool formats (more coming soon)
- Minimal dependencies, targets python 3.7 or higher
- Memory concious - datasets are processed file-by-file instead of loading everything
  in memory (when possible)
- Typed
- Tested with round trip tests to ensure consistency
- MIT license

> **Note**
> Labelformat is a young project, contributions and bug reports are welcome. Please see [Contributing](#contributing) section below.


## Installation

```shell
pip install labelformat
```

## Usage

### CLI

#### Examples

Convert instance segmentation labels from COCO to YOLOv8:
```shell
labelformat convert \
    --task instance-segmentation \
    --input-format coco \
    --input-file coco-labels/train.json \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

Convert object detection labels from KITTI to PascalVOC:
```shell
labelformat convert \
    --task object-detection \
    --input-format kitti \
    --input-folder kitti-labels/labels \
    --category-names cat,dog,fish \
    --images-rel-path ../images \
    --output-format pascalvoc \
    --output-folder pascalvoc-labels
```

Convert object detection labels from Labelbox to Lightly:
```shell
labelformat convert \
    --task object-detection \
    --input-format labelbox \
    --input-file labelbox-labels/export-result.ndjson \
    --category-names cat,dog,fish \
    --output-format lightly \
    --output-folder lightly-labels/annotation-task
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

Please refer to the code for a full list of available classes.

```python
from pathlib import Path
from labelformat.formats import COCOObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Load the input labels
label_input = COCOObjectDetectionInput(
    input_file=Path("coco-labels/train.json")
)
# Convert to output format and save
YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo-labels/data.yaml"),
    output_split="train",
).save(label_input=label_input)
```

### Tutorial

We will walk through in detail how to convert object detection labels from COCO format
to YOLOv8 format and the other way around.

#### Convert Object Detections from COCO to YOLOv8

Let's assume we have `coco.json` in the `coco-labels` directory with following contents:

```json
{
  "info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
  },
  "licenses": [
    {
      "url": "http://creativecommons.org/licenses/by/2.0/",
      "id": 4,
      "name": "Attribution License"
    }
  ],
  "images": [
    {
      "file_name": "image1.jpg",
      "height": 416,
      "width": 640,
      "id": 0,
      "date_captured": "2013-11-18 02:53:27"
    },
    {
      "file_name": "image2.jpg",
      "height": 428,
      "width": 640,
      "id": 1,
      "date_captured": "2016-01-23 13:56:27"
    }
  ],
  "annotations": [
    {
      "area": 421,
      "iscrowd": 0,
      "image_id": 0,
      "bbox": [540, 295, 23, 18],
      "category_id": 2,
      "id": 1
    },
    {
      "area": 695.1853359360001,
      "iscrowd": 0,
      "image_id": 0,
      "bbox": [513, 271, 21, 33],
      "category_id": 0,
      "id": 2
    },
    {
      "area": 27826,
      "iscrowd": 0,
      "image_id": 1,
      "bbox": [268, 63, 94, 295],
      "category_id": 2,
      "id": 16
    }
  ],
  "categories": [
    {
      "supercategory": "animal",
      "id": 0,
      "name": "cat"
    },
    {
      "supercategory": "animal",
      "id": 1,
      "name": "dog"
    },
    {
      "supercategory": "animal",
      "id": 2,
      "name": "fish"
    }
  ]
}
```

Convert it to YOLOv8 format with the following command:

```console
labelformat convert \
  --task object-detection \
  --input-format coco \
  --input-file coco-labels/coco.json \
  --output-format yolov8 \
  --output-file yolo-from-coco-labels/data.yaml \
  --output-split train
```

This creates the following data structure with YOLOv8 labels:

```
yolo-from-coco-labels/
├── data.yaml
└── labels/
    ├── image1.txt
    └── image2.txt
```

The contents of the created files will be:

```
# data.yaml
names:
  0: cat
  1: dog
  2: fish
nc: 3
path: .
train: images

# image1.txt
2 0.86171875 0.7307692307692307 0.0359375 0.04326923076923077
0 0.81796875 0.6911057692307693 0.0328125 0.07932692307692307

# image2.txt
2 0.4921875 0.49182242990654207 0.146875 0.6892523364485982
```

#### Convert Object Detections from YOLOv8 to COCO

Unlike COCO format, YOLO uses relative image coordinates. To convert from YOLO to COCO
we therefore have to provide also input images. We prepare the following folder structure:

```
yolo-labels/
├── data.yaml
├── images/
|   ├── image1.jpg
|   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```

The file contents will be as above. The location of the image folder
is defined in `data.yaml` with the `path` (root path) and `train` field.
Note that YOLO format allows specifying different data folders for
`train`, `val` and `test` data splits, we chose to use `train` for our example.

To convert to COCO run the command below. Note that we specify `--input-split train`:

```console
labelformat convert \
  --task object-detection \
  --input-format yolov8 \
  --input-file yolo-labels/data.yaml \
  --input-split train \
  --output-format coco \
  --output-file coco-from-yolo-labels/coco.json
```

The command will produce `coco-from-yolo-labels/coco.json` with the following contents:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 416
    },
    {
      "id": 1,
      "file_name": "image2.jpg",
      "width": 640,
      "height": 428
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "cat"
    },
    {
      "id": 1,
      "name": "dog"
    },
    {
      "id": 2,
      "name": "fish"
    }
  ],
  "annotations": [
    {
      "image_id": 0,
      "category_id": 2,
      "bbox": [540.0, 295.0, 23.0, 18.0]
    },
    {
      "image_id": 0,
      "category_id": 0,
      "bbox": [513.0, 271.0, 21.0, 33.0]
    },
    {
      "image_id": 1,
      "category_id": 2,
      "bbox": [268.0, 63.0, 94.0, 295.0]
    }
  ]
}
```

Note that converting from COCO to YOLO and back loses some information since
the intermediate format does not store all the fields.

## Contributing

If you encounter a bug or have a feature suggestion we will be happy if you file a GitHub issue.

We also welcome contributions, please submit a PR.

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
