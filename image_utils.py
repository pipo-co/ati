import itertools
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple, Callable, Union, Any

import metadata_repo

import numpy as np
from PIL import Image as PImage

CIRCLE_IMAGE_NAME: str = 'circle.pgm'
SQUARE_IMAGE_NAME: str = 'square.pgm'
RESERVED_IMAGE_NAMES: Tuple[str, ...] = (CIRCLE_IMAGE_NAME, SQUARE_IMAGE_NAME)
COLOR_DEPTH: int = 256
MAX_COLOR: int = COLOR_DEPTH - 1

# (hist, bins)
Hist = Tuple[np.ndarray, np.ndarray]

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

    RED_CHANNEL:    int = 0
    GREEN_CHANNEL:  int = 1
    BLUE_CHANNEL:   int = 2

    def __init__(self, name: str, fmt: ImageFormat, data: np.ndarray, allow_reserved: bool = False):
        if not allow_reserved and name in RESERVED_IMAGE_NAMES:
            raise ValueError(f'name cannot be any of this names: {RESERVED_IMAGE_NAMES}')
        self.name = name
        self.format = fmt
        self.data = data

    def valid_pixel(self, pixel: Tuple[int, int]) -> bool:
        x, y = pixel
        return 0 <= x < self.width and 0 <= y < self.height

    def get_pixel(self, pixel: Tuple[int, int]) -> bool:
        x, y = pixel
        return self.data[y, x]

    def get_channel(self, channel: int) -> np.ndarray:
        return self.data[:, :, channel] if self.channels > 1 else self.data

    def apply_over_channels(self, fn: Callable[[np.ndarray, Any], np.ndarray], *args, **kwargs) -> np.ndarray:
        new_data: np.ndarray
        if self.channels == 1:
            new_data = fn(self.data, *args, **kwargs)
        else:
            new_data = np.empty(self.shape)
            for channel in range(self.channels):
                new_data[:, :, channel] = fn(self.get_channel(channel), *args, **kwargs)

        return new_data

    def get_histograms(self) -> Union[Tuple[Hist], Tuple[Hist, Hist, Hist]]:
        if self.channels == 1:
            return channel_histogram(self.data),
        else:
            return (
                channel_histogram(self.get_channel(Image.RED_CHANNEL)),
                channel_histogram(self.get_channel(Image.GREEN_CHANNEL)),
                channel_histogram(self.get_channel(Image.BLUE_CHANNEL))
            )

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def type(self) -> int:
        return self.data.dtype

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
    normalized_data = normalize(image.data)
    if image.channels == 1:
        return _grayscale_to_rgba(normalized_data)
    elif image.channels == 3:
        return _color_to_rgba(normalized_data)

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
        data = data.reshape((metadata.height, metadata.width))
    else:
        data = np.asarray(PImage.open(path), dtype=np.uint8) # noqa

    return Image(name, fmt, data.astype(np.float64))

def save_image(image: Image, dir_path: str) -> None:
    normalized_data = normalize(image.data)
    path = os.path.join(dir_path, strip_extension(image.name)) + image.format.to_extension()
    if image.format == ImageFormat.RAW:
        # Write bytes from data
        with open(path, 'wb') as fp:
            for b in normalized_data.flatten():
                fp.write(b)
        # Write metadata
        metadata_repo.persist_image_metadata(image.name, image.width, image.height)
    else:
        PImage.fromarray(normalized_data).save(path)


CREATED_IMAGE_LEN: int = 200
CIRCLE_RADIUS: int = 100
def create_circle_image() -> Image:
    mask = create_circular_mask(CREATED_IMAGE_LEN, CREATED_IMAGE_LEN, radius=CIRCLE_RADIUS)
    data = np.zeros((CREATED_IMAGE_LEN, CREATED_IMAGE_LEN), dtype=np.float64)
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
    data = np.zeros((CREATED_IMAGE_LEN, CREATED_IMAGE_LEN), dtype=np.float64)
    data[min_square:max_square, min_square:max_square] = 255

    return Image(SQUARE_IMAGE_NAME, ImageFormat.PGM, data, allow_reserved=True)

def add_images(first_img: Image, second_img: Image) -> np.ndarray:
    return np.add(first_img.data, second_img.data)

def sub_images(first_img: Image, second_img: Image) -> np.ndarray:
    return np.subtract(first_img.data, second_img.data)

def multiply_images(first_img: Image, second_img: Image) -> np.ndarray:
    return np.multiply(first_img.data, second_img.data)

# Normalizes to uint8 ndarray
def normalize(data: np.ndarray, as_type=np.uint8) -> np.ndarray:
    if data.dtype == np.uint8:
        return data.astype(as_type, copy=False)
    elif np.can_cast(data.dtype, np.uint8, casting='safe'):
        return data.astype(as_type, copy=False)
    else:
        rng = data.max() - data.min()
        if rng == 0:
            return np.zeros(data.shape, as_type)
        else:
            amin = data.min()
            ret = (data - amin) * 255 // rng
            return ret.astype(as_type, copy=False)

def power_function(img: Image, gamma: float) -> np.ndarray:
    c = MAX_COLOR**(1 - gamma)
    return c*(img.data**gamma)

def negate(img: Image) -> np.ndarray:
    return np.array([MAX_COLOR - xi for xi in img.data])

def to_binary(img: Image, umb: int) -> np.ndarray:
    return img.apply_over_channels(channel_to_binary, umb)

# TODO(tobi, nacho): Vectorizar
def channel_to_binary(channel: np.ndarray, umb: int) -> np.ndarray:
    shape = np.shape(channel)
    new_arr = np.array([MAX_COLOR if xi >= umb else 0 for xi in channel.flatten()])
    return np.reshape(new_arr, shape)

def channel_histogram(channel: np.ndarray) -> Hist:
    channel = normalize(channel, np.float64)
    hist, bins = np.histogram(channel.flatten(), bins=COLOR_DEPTH, range=(0, COLOR_DEPTH))
    return hist / channel.size, bins

def channel_equalization(channel: np.ndarray)  -> np.ndarray:
    channel = normalize(channel, np.float64)
    normed_hist, bins = channel_histogram(channel)
    s = normed_hist.cumsum()
    masked_s = np.ma.masked_equal(s, 0)
    masked_min = masked_s.min()
    masked_s = (masked_s - masked_min) * MAX_COLOR / (masked_s.max() - masked_min)
    s = np.ma.filled(masked_s, 0)
    return s[channel]

def equalize(image: Image) -> np.ndarray:
    return image.apply_over_channels(channel_equalization)
