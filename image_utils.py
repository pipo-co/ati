import itertools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

import metadata_repo as metadata_repo

import numpy as np
from PIL import Image as PImage

circle_image_path: str = '/home/fpannunzio/ATI/ati/images/circle.pgm'
square_image_path: str = '/home/fpannunzio/ATI/ati/images/square.pgm'

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
def load_image(path: str) -> Image:
    name = Image.name_from_path(path)
    ext = get_extension(path)
    if ext == '.RAW':
        metadata = metadata_repo.get_metadata(name)
        npimg = np.fromfile(path, dtype=np.uint8)
        imageSize = (metadata.width, metadata.height)
        npimg = npimg.reshape(imageSize)
        return Image(name, ImageFormat.from_extension(ext), npimg) 

    return Image(name, ImageFormat.from_extension(ext), np.asarray(PImage.open(path)))

# TODO: Add raw type support
def save_image(image: Image, dir_path: str) -> None:
    path = os.path.join(dir_path, strip_extension(image.name)) + image.format.to_extension()
    if get_extension(path) == ".raw":
        split_name = os.path.splitext(os.path.basename(path))
        PImage.fromarray(image.data/1023.0).save(split_name[0] + ".tiff") #Polemico
        return
    PImage.fromarray(image.data).save(path)


def load_metadata(path: str) -> None:
    metadata_repo.load_metadata(path)

def create_circular_image() -> None:
    mask = create_circular_mask(200, 200, radius=100)
    array = np.zeros((200, 200), dtype=np.uint8)
    array[mask] = 255
    PImage.fromarray(array).save(circle_image_path)

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def create_square_image() -> None:
    array = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            if i >= 20 and i <= 180:
                if j >= 20 and j<=180: 
                    array[i][j] = 255

    PImage.fromarray(array).save(square_image_path)

def sum_images(first_img: Image, second_img: Image) -> np.ndarray:
    array = np.add(first_img.data.astype(np.uint) , second_img.data.astype(np.uint))
    new_array = normalize(array)
    print(first_img.data[100,100], second_img.data[100,100], new_array[100,100])
    return new_array

def normalize(arr: np.ndarray) -> np.ndarray:
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng