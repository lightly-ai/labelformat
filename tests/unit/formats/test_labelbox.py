import pytest

from labelformat.formats.labelbox import _has_illegal_char, _image_from_data_row
from labelformat.types import ParseError


class TestLabelboxFunctions:
    def test_has_illegal_char(self) -> None:
        assert _has_illegal_char("filename/with/slash")
        assert _has_illegal_char("filename\\with\\backslash")
        assert _has_illegal_char("filename:with:colon")
        assert not _has_illegal_char("valid_filename")

    def test_image_from_data_row_valid(self) -> None:
        data_row = {
            "data_row": {"global_key": "image123", "id": "123"},
            "media_attributes": {"width": 800, "height": 600},
        }
        image = _image_from_data_row(1, data_row, "global_key")
        assert image.id == 1
        assert image.filename == "image123"
        assert image.width == 800
        assert image.height == 600

    def test_image_from_data_row_illegal_char(self) -> None:
        data_row = {
            "data_row": {"global_key": "image/123", "id": "123"},
            "media_attributes": {"width": 800, "height": 600},
        }
        with pytest.raises(ParseError):
            _image_from_data_row(1, data_row, "global_key")

    def test_image_from_data_row_key_not_found(self) -> None:
        data_row = {
            "data_row": {"id": "123"},
            "media_attributes": {"width": 800, "height": 600},
        }
        with pytest.raises(ParseError):
            _image_from_data_row(1, data_row, "global_key")
