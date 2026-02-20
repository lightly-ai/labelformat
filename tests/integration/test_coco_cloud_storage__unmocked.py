import pytest
from fsspec.core import url_to_fs

from labelformat.formats.coco import COCOObjectDetectionInput

COCO_S3_ANNOTATION_FILE = (
    "s3://studio-test-datasets-eu/coco_subset_128_images/instances_train2017.json"
)
COCO_S3_IMAGES_DIR = "s3://studio-test-datasets-eu/coco_subset_128_images/images/"


@pytest.mark.skip(reason="Requires access to S3 dataset")
def test_coco_od_inputs_read_from_s3__unmocked() -> None:
    object_detection_input = COCOObjectDetectionInput(
        input_file=COCO_S3_ANNOTATION_FILE
    )

    object_detection_categories = list(object_detection_input.get_categories())
    object_detection_images = list(object_detection_input.get_images())
    object_detection_labels = list(object_detection_input.get_labels())

    assert len(object_detection_categories) == 80
    assert len(object_detection_images) == 128
    assert len(object_detection_labels) == 128
    assert sum(len(label.objects) for label in object_detection_labels) == 900

    # Assert that the images found in the annotations can be accessed in S3
    for image in object_detection_images:
        image_path = COCO_S3_IMAGES_DIR + image.filename
        fs, fs_path = url_to_fs(image_path)
        assert fs.exists(fs_path)
