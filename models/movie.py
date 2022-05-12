import os
import abc
from dataclasses import dataclass
from typing import List, Callable

from .image import Image, ImageTransformation


@dataclass
class MovieTransformation(ImageTransformation):
    inductive_handle: Callable[[Image], Image]

    def __init__(self, name: str, inductive_handle: Callable[[Image], Image], **kwargs):
        super().__init__(name, **kwargs)
        self.inductive_handle = inductive_handle


@dataclass
class Movie(abc.ABC):
    name:               str
    frames:             List[str]
    current_frame:      int

    def __init__(self, name: str, frames: List[str]) -> None:
        if not frames:
            raise ValueError('At least 1 frame is required for movie')

        self.name               = name
        self.frames             = frames
        self.current_frame      = 0

    def __len__(self):
        return len(self.frames)

    def on_first_frame(self) -> bool:
        return self.current_frame == 0

    def on_last_frame(self) -> bool:
        return self.current_frame == len(self.frames) - 1

    @property
    @abc.abstractmethod
    def transformations(self) -> List[MovieTransformation]:
        pass

    @property
    def current_frame_name(self) -> str:
        return self.get_frame_name(self.current_frame)

    @abc.abstractmethod
    def get_frame_name(self, frame: int) -> str:
        pass

@dataclass
class RootMovie(Movie):
    base_path: str

    def __init__(self, name: str, frames: List[str], base_path: str) -> None:
        super().__init__(name, frames)

        self.base_path = base_path

    @property
    def transformations(self) -> List[MovieTransformation]:
        return []

    # Override
    def get_frame_name(self, frame: int) -> str:
        return Image.name_from_path(self.frames[frame])

    @property
    def current_frame_path(self) -> str:
        return os.path.join(self.base_path, self.frames[self.current_frame])

@dataclass
class TransformedMovie(Movie):
    base_movie:         str
    transformations:    List[MovieTransformation]

    def __init__(self, new_name: str, base_movie: Movie, transformation: MovieTransformation) -> None:
        new_frames = [f'{new_name}_{i}' for i in range(len(self.frames))]
        super().__init__(new_name, new_frames)

        self.base_movie         = base_movie.name
        self.transformations    = base_movie.transformations + [transformation]

    # Override
    def get_frame_name(self, frame: int) -> str:
        return self.frames[frame]
