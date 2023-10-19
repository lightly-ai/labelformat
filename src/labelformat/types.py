from typing import Any, Dict

JsonDict = Dict[str, Any]  # type: ignore[misc]


class ParseError(Exception):
    pass
