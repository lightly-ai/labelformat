from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Video:
    id: int
    filenames: List[str]
    width: int
    height: int
    length: int
