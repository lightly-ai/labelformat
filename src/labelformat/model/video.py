from dataclasses import dataclass


@dataclass(frozen=True)
class Video:
    id: int
    filename: str
    width: int
    height: int
    number_of_frames: int
    # TODO (Jonas, 01/2026): Add list of frames
