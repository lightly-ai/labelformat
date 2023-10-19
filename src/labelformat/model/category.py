from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Category:
    id: int
    name: str
