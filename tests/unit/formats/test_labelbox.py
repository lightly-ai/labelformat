import pytest

from labelformat.formats import labelbox
from labelformat.formats.labelbox import FilenameKeyOption
from labelformat.types import ParseError


def test_has_illegal_char() -> None:
    assert labelbox._has_illegal_char("filename/with/slash")
    assert labelbox._has_illegal_char("filename\\with\\backslash")
    assert labelbox._has_illegal_char("filename:with:colon")
    assert not labelbox._has_illegal_char("valid_filename")


def test_image_from_data_row__valid() -> None:
    data_row = {
        "data_row": {"global_key": "image123", "id": "123"},
        "media_attributes": {"width": 800, "height": 600},
    }
    image = labelbox._image_from_data_row(
        image_id=1, data_row=data_row, filename_key=FilenameKeyOption.GLOBAL_KEY
    )
    assert image.id == 1
    assert image.filename == "image123"
    assert image.width == 800
    assert image.height == 600


def test_image_from_data_row__illegal_char() -> None:
    data_row = {
        "data_row": {"global_key": "image/123", "id": "123"},
        "media_attributes": {"width": 800, "height": 600},
    }
    with pytest.raises(ParseError):
        labelbox._image_from_data_row(
            image_id=1, data_row=data_row, filename_key=FilenameKeyOption.GLOBAL_KEY
        )


def test_image_from_data_row__key_not_found() -> None:
    data_row = {
        "data_row": {"id": "123"},
        "media_attributes": {"width": 800, "height": 600},
    }
    with pytest.raises(ParseError):
        labelbox._image_from_data_row(
            image_id=1, data_row=data_row, filename_key=FilenameKeyOption.GLOBAL_KEY
        )
