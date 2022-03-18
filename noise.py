import functools
from enum import Enum
from typing import Callable, List

import numpy as np

import rng
from image_utils import Image, MAX_COLOR

NoiseSupplier = Callable[[], float]

class NoiseType(Enum):
    ADDITIVE        = functools.partial(lambda val, noise: val + noise)
    MULTIPLICATIVE  = functools.partial(lambda val, noise: val * noise)

    @classmethod
    def names(cls) -> List[str]:
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_name(cls, name: str) -> 'NoiseType':
        name = name.upper()
        if name not in cls.__members__:
            raise ValueError(f'"{name.capitalize()}" is not a supported noise type')
        return cls[name]

    def __call__(self, val: int, noise: float) -> float:
        return self.value(val, noise)

def pollute(img: Image, noise_supplier: NoiseSupplier, noise_type: NoiseType, percentage: int) -> np.ndarray:
    return img.apply_over_channels(pollute_channel, noise_supplier, noise_type, percentage)

# TODO(tobi, nacho): Vectorizar
def pollute_channel(channel: np.ndarray, noise_supplier: NoiseSupplier, noise_type: NoiseType, percentage: int) -> np.ndarray:
    p = percentage / 100
    shape = np.shape(channel)
    ret = np.array([(noise_type(xi, int(noise_supplier() * MAX_COLOR))) if rng.random() < p else xi for xi in channel.flatten()])
    return np.reshape(ret, shape)

def salt(img: Image, percentage: int) -> np.ndarray:
    return img.apply_over_channels(salt_channel, percentage)

def salted_pixel(xi: int, p: float) -> int:
    obs = rng.random()
    if obs > 1 - p:
        return MAX_COLOR
    elif obs < p:
        return 0
    else:
        return xi

# TODO(tobi, nacho): Vectorizar
def salt_channel(channel: np.ndarray, percentage: int) -> np.ndarray:
    p = percentage / 100
    shape = np.shape(channel)
    new_arr = np.array([salted_pixel(xi, p) for xi in channel.flatten()], dtype=np.uint8)
    return np.reshape(new_arr, shape)
