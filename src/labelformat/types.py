from pathlib import Path
from typing import Any, Dict, Union

JsonDict = Dict[str, Any]


class ParseError(Exception):
    pass


PathLike = Union[str, Path]
