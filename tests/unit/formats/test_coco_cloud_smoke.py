from labelformat.formats.coco import (
    COCOInstanceSegmentationInput,
    COCOObjectDetectionInput,
)

COCO_S3_ANNOTATION_FILE = (
    "s3://studio-test-datasets-eu/coco_subset_128_images/instances_train2017.json"
)
COCO_S3_IMAGES_DIR = "s3://studio-test-datasets-eu/coco_subset_128_images/images/"


def test_coco_inputs_read_from_s3() -> None:
    object_detection_input = COCOObjectDetectionInput(
        input_file=COCO_S3_ANNOTATION_FILE
    )
    instance_segmentation_input = COCOInstanceSegmentationInput(
        input_file=COCO_S3_ANNOTATION_FILE
    )

    object_detection_categories = list(object_detection_input.get_categories())
    object_detection_images = list(object_detection_input.get_images())
    object_detection_labels = list(object_detection_input.get_labels())

    instance_segmentation_categories = list(
        instance_segmentation_input.get_categories()
    )
    instance_segmentation_images = list(instance_segmentation_input.get_images())
    instance_segmentation_labels = list(instance_segmentation_input.get_labels())

    assert len(object_detection_categories) > 0
    assert len(object_detection_images) > 0
    assert len(object_detection_labels) > 0
    assert sum(len(label.objects) for label in object_detection_labels) > 0

    assert len(instance_segmentation_categories) > 0
    assert len(instance_segmentation_images) > 0
    assert len(instance_segmentation_labels) > 0
    assert sum(len(label.objects) for label in instance_segmentation_labels) > 0

    assert all(
        (COCO_S3_IMAGES_DIR + image.filename).startswith(COCO_S3_IMAGES_DIR)
        for image in object_detection_images
    )
