import itertools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

import metadata_repo

import numpy as np
from PIL import ImageChops, Image as PImage

CIRCLE_IMAGE_NAME: str = 'circle.pgm'
SQUARE_IMAGE_NAME: str = 'square.pgm'
RESERVED_IMAGE_NAMES: Tuple[str, ...] = (CIRCLE_IMAGE_NAME, SQUARE_IMAGE_NAME)

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

    def __init__(self, name: str, fmt: ImageFormat, data: np.ndarray, allow_reserved: bool = False):
        if not allow_reserved and name in RESERVED_IMAGE_NAMES:
            raise ValueError(f'name cannot be any of this names: {RESERVED_IMAGE_NAMES}')
        self.name = name
        self.format = fmt
        self.data = data

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
    fmt = ImageFormat.from_extension(get_extension(name))
    data: np.ndarray
    if fmt == ImageFormat.RAW:
        metadata = metadata_repo.get_metadata(name)
        data = np.fromfile(path, dtype=np.uint8)
        data = data.reshape((metadata.width, metadata.height))
    else:
        data = np.asarray(PImage.open(path), dtype=np.uint8)

    return Image(name, fmt, data)

def save_image(image: Image, dir_path: str) -> None:
    path = os.path.join(dir_path, strip_extension(image.name)) + image.format.to_extension()
    if image.format == ImageFormat.RAW:
        # Write bytes from data
        with open(path, 'wb') as fp:
            for b in image.data.flatten():
                fp.write(b)
        # Write metadata
        metadata_repo.persist_image_metadata(image.name, image.width, image.height)
    else:
        PImage.fromarray(image.data).save(path)


CREATED_IMAGE_LEN:  int = 200
CIRCLE_RADIUS: int = 100
def create_circle_image() -> Image:
    mask = create_circular_mask(CREATED_IMAGE_LEN, CREATED_IMAGE_LEN, radius=CIRCLE_RADIUS)
    data = np.zeros((CREATED_IMAGE_LEN, CREATED_IMAGE_LEN), dtype=np.uint8)
    data[mask] = 255
    return Image(CIRCLE_IMAGE_NAME, ImageFormat.PGM, data, allow_reserved=True)

# https://stackoverflow.com/a/44874588
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


SQUARE_LEN: int = 160
def create_square_image() -> Image:
    diff = (CREATED_IMAGE_LEN - SQUARE_LEN) // 2
    min_square = diff
    max_square = CREATED_IMAGE_LEN - diff
    data = np.zeros((CREATED_IMAGE_LEN, CREATED_IMAGE_LEN), dtype=np.uint8)
    for i in range(CREATED_IMAGE_LEN):
        for j in range(CREATED_IMAGE_LEN):
            if min_square <= i <= max_square:
                if min_square <= j <= max_square:
                    data[i][j] = 255

    return Image(SQUARE_IMAGE_NAME, ImageFormat.PGM, data, allow_reserved=True)

def sum_images(first_img: Image, second_img: Image) -> np.ndarray:
    array = np.add(first_img.data.astype(np.uint), second_img.data.astype(np.uint))
    new_array = normalize(array)
    return new_array

def sub_images(first_img: Image, second_img: Image) -> np.ndarray:
    array = np.subtract(first_img.data.astype(np.uint), second_img.data.astype(np.uint))
    new_array = normalize(array)
    return new_array

def multiply_images(first_img: Image, second_img: Image) -> np.ndarray:
    # creating a image1 object
    print(first_img.name, second_img.name)
    im1 = PImage.open("images/" + first_img.name)

    # creating a image2 object
    im2 = PImage.open("images/" + second_img.name)
    print(im1, im2)

    return ImageChops.multiply(im1, im2)

        
def normalize(arr: np.ndarray) -> np.ndarray:
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin)*255//rng
