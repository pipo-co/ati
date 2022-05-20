import os
import abc
from dataclasses import dataclass
from typing import List, Callable, Dict, Any

from .image import Image, ImageTransformation
from .path_utils import get_extension


@dataclass
class MovieTransformation(ImageTransformation):
    inductive_handle: Callable[[str, int, Image, Image], Image]

    @classmethod
    def from_img_tr(cls, image_transformation: ImageTransformation, inductive_handle: Callable[[str, int, Image, Image], Image]) -> 'MovieTransformation':
        return cls(image_transformation.name, inductive_handle, image_transformation.major_inputs, image_transformation.minor_inputs)

    def __init__(self, name: str, inductive_handle: Callable[[str, int, Image, Image], Image], major_inputs: Dict[str, Any], minor_inputs: Dict[str, Any]):
        super().__init__(name, major_inputs, minor_inputs, [])
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

    def inc_frame(self) -> None:
        self.current_frame += 1

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

    def get_frame_path(self, frame: int) -> str:
        return os.path.join(self.base_path, self.frames[frame])

@dataclass
class TransformedMovie(Movie):
    base_movie:         str
    transformations:    List[MovieTransformation]

    def __init__(self, new_name: str, base_movie: Movie, transformation: MovieTransformation) -> None:
        ext = get_extension(base_movie.get_frame_name(0))
        new_frames = [f'{new_name}_{i}{ext}' for i in range(len(base_movie.frames))]
        super().__init__(new_name, new_frames)

        self.base_movie         = base_movie.name
        self.transformations    = base_movie.transformations + [transformation]

    @property
    def last_transformation(self) -> MovieTransformation:
        return self.transformations[-1]

    # Override
    def get_frame_name(self, frame: int) -> str:
        return self.frames[frame]

    # Override
    def transformations(self) -> List[MovieTransformation]:
        return self.transformations
