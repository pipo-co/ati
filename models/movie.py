import os
from dataclasses import dataclass, field
from typing import List

from .image import Image
from .transformation import Transformation


@dataclass
class Movie:
    name:               str
    base_path:          str
    frames:             List[str]
    current_frame:      int
    transformations:    List[Transformation]

    def __init__(self, name: str, base_path: str, frames: List[str]) -> None:
        if not frames:
            raise ValueError('At least 1 frame is required for movie')

        self.name               = name
        self.base_path          = base_path
        self.frames             = frames
        self.current_frame      = 0
        self.transformations    = []

    def __len__(self):
        return len(self.frames)

    def on_first_frame(self) -> bool:
        return self.current_frame == 0

    def on_last_frame(self) -> bool:
        return self.current_frame == len(self.frames) - 1

    @property
    def current_frame_name(self) -> str:
        return Image.name_from_path(self.frames[self.current_frame])

    @property
    def current_frame_path(self) -> str:
        return os.path.join(self.base_path, self.frames[self.current_frame])

