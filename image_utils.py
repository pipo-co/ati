import itertools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

import numpy as np
from PIL import Image as PImage

raw_images_metadata_path: str = 'images/raw_metadata.csv'

class ImageFormat(Enum):
    PGM     = 'pgm'
    PPM     = 'ppm'
    JPEG    = 'jpeg'
    JPG     = 'jpg'
    RAW     = 'raw'

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def from_str(cls, fmt):
        if fmt not in ImageFormat.values():
            raise ValueError(f'"{fmt}" is not a supported image format')
        return cls(fmt)

    @classmethod
    def from_extension(cls, ext):
        return cls.from_str((ext[1:] if len(ext) > 0 and ext[0] == '.' else ext).lower())

    def to_extension(self) -> str:
        return '.' + self.value

@dataclass
class Image:
    name:   str
    format: ImageFormat
    data:   np.ndarray

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @property
    def width(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        shape = self.data.shape
        return 1 if len(shape) == 2 else shape[2]

    @staticmethod
    def name_from_path(path: str) -> str:
        split_name = os.path.splitext(os.path.basename(path))
        return split_name[0] + split_name[1].lower()

def valid_image_formats() -> Iterable[str]:
    formats = list(map(lambda fmt: fmt.to_extension(), ImageFormat))
    return itertools.chain(formats, map(lambda f: f.upper(), formats))

def _grayscale_to_rgba(data: np.ndarray) -> np.ndarray:
    # Tenemos que repetir el valor por cada canal de color, y agregar 1 por el canal del alpha
    return _color_to_rgba(np.repeat(data.reshape((*data.shape, 1)), 3, axis=2))

def _color_to_rgba(data: np.ndarray) -> np.ndarray:
    # Solamente hace falta agregar el canal del alpha (que siempre es 1)
    return np.insert(data, 3, 255, axis=2).flatten() / 255

def image_to_rgba_array(image: Image) -> np.ndarray:
    if image.channels == 1:
        return _grayscale_to_rgba(image.data)
    elif image.channels == 3:
        return _color_to_rgba(image.data)

def get_extension(path: str) -> str:
    return os.path.splitext(path)[1]

def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]

def append_to_filename(filename: str, s: str) -> str:
    split = os.path.splitext(filename)
    return split[0] + s + split[1]

# height x width x channel
# TODO: Add raw type support
def load_image(path: str) -> Image:
    name = Image.name_from_path(path)
    ext = get_extension(path)

    return Image(name, ImageFormat.from_extension(ext), np.asarray(PImage.open(path)))

# TODO: Add raw type support
def save_image(image: Image, dir_path: str) -> None:
    path = os.path.join(dir_path, strip_extension(image.name)) + image.format.to_extension()
    PImage.fromarray(image.data).save(path)
