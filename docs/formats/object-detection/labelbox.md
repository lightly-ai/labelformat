# Labelbox Object Detection Format

## Overview
Labelbox uses NDJSON (Newline Delimited JSON) format for label exports, where each line represents a single image and its annotations. The format supports object detection through bounding boxes.
While Labelformat currently supports Labelbox as an input-only format, you can find the complete format specification in the [Labelbox documentation](https://docs.labelbox.com/reference/label-export).

## Specification of Labelbox Detection Format
```
dataset/
└── export-result.ndjson
```

Each line in the NDJSON file contains a complete JSON object with three main sections:

- `data_row`: Contains image metadata (id, filename, external references)
- `media_attributes`: Image dimensions
- `projects`: Contains the actual annotations

## Label Format
Each annotation line follows this structure:
```json
{
  "data_row": {
    "id": "data_row_id",
    "global_key": "image1.jpg",
    "external_id": "image1.jpg"
  },
  "media_attributes": {
    "width": 640,
    "height": 480
  },
  "projects": {
    "project_id": {
      "labels": [{
        "annotations": {
          "objects": [{
            "name": "cat",
            "annotation_kind": "ImageBoundingBox",
            "bounding_box": {
              "top": 100,
              "left": 200,
              "width": 50,
              "height": 30
            }
          }]
        }
      }]
    }
  }
}
```

## Converting from Labelbox Format
Labelbox format can be converted to other formats using labelformat. Here's an example converting to YOLOv8:

```bash
labelformat convert \
    --task object-detection \
    --input-format labelbox \
    --input-file labelbox-labels/export-result.ndjson \
    --category-names cat,dog,fish \
    --output-format yolov8 \
    --output-file yolo-labels/data.yaml \
    --output-split train
```

### Important Parameters
- `--category-names`: Required list of category names (comma-separated)
- `--filename-key`: Which key to use as filename (options: global_key, external_id, id; default: global_key)

## Format Details

### Bounding Box Format
- Uses absolute pixel coordinates
- Format: `{top, left, width, height}`
- Origin: Top-left corner of the image

### Limitations
- Currently supports single project exports only
- Video annotations are not supported
- Only `ImageBoundingBox` annotation types are processed

## Example
```json
{"data_row":{"id":"ckz...","global_key":"image1.jpg","external_id":"img_1"},"media_attributes":{"width":640,"height":480},"projects":{"proj_123":{"labels":[{"annotations":{"objects":[{"name":"cat","annotation_kind":"ImageBoundingBox","bounding_box":{"top":100,"left":200,"width":50,"height":30}}]}}]}}}
{"data_row":{"id":"ckz...","global_key":"image2.jpg","external_id":"img_2"},"media_attributes":{"width":640,"height":480},"projects":{"proj_123":{"labels":[{"annotations":{"objects":[{"name":"dog","annotation_kind":"ImageBoundingBox","bounding_box":{"top":150,"left":300,"width":60,"height":40}}]}}]}}}
```

Note: This format is supported for input only in labelformat. 