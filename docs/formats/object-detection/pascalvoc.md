# PascalVOC Object Detection Format

## Overview
PascalVOC (Visual Object Classes) is a widely used format for object detection tasks, introduced in the seminal paper ["The PASCAL Visual Object Classes (VOC) Challenge"](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf) by Everingham et al. It stores annotations in XML files, with one XML file per image containing bounding box coordinates and class labels. The complete format specification is available in the [PascalVOC development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit).

## Specification
Each XML annotation file contains:
- Image metadata (filename, size, etc.)
- List of objects, each with:
  - Class name (string, allows spaces, e.g., "traffic light" or "stop sign")
  - Bounding box coordinates as integer pixel values:
    - xmin: left-most pixel coordinate
    - ymin: top-most pixel coordinate
    - xmax: right-most pixel coordinate
    - ymax: bottom-most pixel coordinate
  - Optional attributes (difficult, truncated, occluded)

## Directory Structure
```
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── annotations/
    ├── image1.xml
    └── image2.xml
```

## Example Annotation
```xml
<annotation>
    <folder>images</folder>
    <filename>image1.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>cat</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
```

## Format Details
- Coordinates are in absolute pixel values (not normalized)
- Bounding boxes use XYXY format (xmin, ymin, xmax, ymax)
- Each object can have optional attributes:
  - `difficult`: Indicates hard to recognize objects
  - `truncated`: Indicates objects partially outside the image
  - `occluded`: Indicates partially obscured objects

## Converting with Labelformat

### COCO to PascalVOC
```bash
labelformat convert \
    --task object-detection \
    --input-format coco \
    --input-file coco-labels/annotations.json \
    --output-format pascalvoc \
    --output-folder pascalvoc-labels
```

### PascalVOC to COCO
```bash
labelformat convert \
    --task object-detection \
    --input-format pascalvoc \
    --input-folder pascalvoc-labels \
    --category-names cat,dog,fish \
    --output-format coco \
    --output-file coco-labels/annotations.json
```

### Required Arguments
- For input:
  - `--input-folder`: Directory containing PascalVOC XML files
  - `--category-names`: Comma-separated list of category names (e.g., 'dog,cat')
- For output:
  - `--output-folder`: Directory to save generated XML files

## References
- [Original PascalVOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Format Documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf) 