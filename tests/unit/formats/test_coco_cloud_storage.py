import io
import json

from pytest_mock import MockerFixture

from labelformat.formats.coco import COCOObjectDetectionInput

COCO_S3_ANNOTATION_FILE = "s3://some_bucket/some_file.json"
MOCK_COCO_PAYLOAD = {
    "categories": [{"id": 1, "name": "cat"}],
    "images": [],
    "annotations": [],
}


def test_coco_od_inputs_read_from_s3__mocked(mocker: MockerFixture) -> None:
    mock_open = mocker.patch(
        "labelformat.formats.coco.fsspec.open",
        return_value=io.StringIO(json.dumps(MOCK_COCO_PAYLOAD)),
    )
    object_detection_input = COCOObjectDetectionInput(
        input_file=COCO_S3_ANNOTATION_FILE
    )
    object_detection_categories = list(object_detection_input.get_categories())

    assert len(object_detection_categories) == 1
    assert object_detection_categories[0].name == "cat"
    mock_open.assert_called_once_with(COCO_S3_ANNOTATION_FILE, mode="r")
