import io
import json
from typing import Any, Dict

import fsspec.core
from pytest_mock import MockerFixture

from labelformat.formats.coco import COCOObjectDetectionInput

COCO_S3_ANNOTATION_FILE = "s3://some_bucket/some_file.json"
COCO_S3_IMAGES_DIR = "s3://some_bucket/some_images/"

MOCK_COCO_PAYLOAD: Dict[str, Any] = {
    "categories": [
        {"id": 1, "name": "cat"},
        {"id": 2, "name": "dog"},
        {"id": 3, "name": "cow"},
    ],
    "images": [
        {"id": 1, "file_name": "image_001.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "image_002.jpg", "width": 800, "height": 600},
        {"id": 3, "file_name": "image_003.jpg", "width": 1024, "height": 768},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [15.0, 25.0, 35.0, 45.0]},
        {"id": 3, "image_id": 2, "category_id": 3, "bbox": [50.0, 60.0, 70.0, 80.0]},
        {"id": 4, "image_id": 3, "category_id": 1, "bbox": [5.0, 10.0, 15.0, 20.0]},
    ],
}


def test_coco_od_inputs_read_from_s3_mocked(mocker: MockerFixture) -> None:
    mock_open = mocker.patch(
        "labelformat.formats.coco.fsspec.open",
        return_value=io.StringIO(json.dumps(MOCK_COCO_PAYLOAD)),
    )
    mock_fs = mocker.MagicMock()
    mock_fs.exists.return_value = True

    def mock_url_to_fs(path: str) -> Any:
        assert path.startswith(COCO_S3_IMAGES_DIR)
        return mock_fs, path

    mocked_url_to_fs = mocker.patch.object(
        fsspec.core, "url_to_fs", side_effect=mock_url_to_fs
    )

    object_detection_input = COCOObjectDetectionInput(
        input_file=COCO_S3_ANNOTATION_FILE
    )

    object_detection_categories = list(object_detection_input.get_categories())
    object_detection_images = list(object_detection_input.get_images())
    object_detection_labels = list(object_detection_input.get_labels())

    assert len(object_detection_categories) == 3
    assert len(object_detection_images) == 3
    assert len(object_detection_labels) == 3
    assert sum(len(label.objects) for label in object_detection_labels) == 4
    mock_open.assert_called_once_with(COCO_S3_ANNOTATION_FILE, mode="r")

    for image in object_detection_images:
        image_path = COCO_S3_IMAGES_DIR + image.filename
        fs, fs_path = fsspec.core.url_to_fs(image_path)
        assert fs.exists(fs_path)

    assert mocked_url_to_fs.call_count == len(object_detection_images)
    assert mock_fs.exists.call_count == len(object_detection_images)
