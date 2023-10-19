from dataclasses import dataclass


@dataclass(frozen=True)
class Image:
    id: int
    filename: str
    width: int
    height: int
