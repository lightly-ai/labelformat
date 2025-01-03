# Lightly Object Detection Format

## Overview
The Lightly format is designed for efficient handling of object detection predictions in machine learning workflows. It provides a straightforward structure that's easy to parse and generate. For detailed information about the prediction format, refer to the [Lightly AI documentation](https://docs.lightly.ai/docs/prediction-format#prediction-format).

## Specification of Lightly Detection Format
The format uses a JSON file per image containing:
- `file_name`: Name of the image file
- `predictions`: List of object detections
  - `category_id`: Integer ID of the object category
  - `bbox`: List of [x, y, width, height] in absolute pixel coordinates
  - `score`: Optional confidence score (0-1)

## File Structure
```
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── predictions/
    ├── image1.json
    └── image2.json
```

## Example
```json
{
  "file_name": "image1.jpg",
  "predictions": [
    {
      "category_id": 0,
      "bbox": [100, 200, 50, 30],
      "score": 0.95
    },
    {
      "category_id": 1,
      "bbox": [300, 400, 80, 60],
      "score": 0.87
    }
  ]
}
``` 