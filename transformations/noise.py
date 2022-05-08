import functools
from enum import Enum
from typing import Callable, List

import numpy as np

import rng
from models.image import Image, MAX_COLOR

NoiseSupplier = Callable[[int], float]

class Type(Enum):
    ADDITIVE        = functools.partial(lambda vals, noise: vals + noise * MAX_COLOR)
    MULTIPLICATIVE  = functools.partial(lambda vals, noise: vals * noise)

    @classmethod
    def names(cls) -> List[str]:
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_name(cls, name: str) -> 'Type':
        name = name.upper()
        if name not in cls.__members__:
            raise ValueError(f'"{name.title()}" is not a supported noise type')
        return cls[name]

    def __call__(self, vals: np.ndarray, noise: float) -> np.ndarray:
        return self.value(vals, noise)

def pollute(img: Image, noise_supplier: NoiseSupplier, noise_type: Type, percentage: int) -> np.ndarray:
    return img.apply_over_channels(pollute_channel, noise_supplier, noise_type, percentage)

def pollute_channel(channel: np.ndarray, noise_supplier: NoiseSupplier, noise_type: Type, percentage: int) -> np.ndarray:
    p = percentage / 100
    n = int(channel.size * p)
    shape = np.shape(channel)
    indices = rng.rng.choice(channel.size, n, replace=False) 
    ret = channel.flatten()
    noise = noise_supplier(n)
    ret[indices] = noise_type(ret[indices], noise)
    return np.reshape(ret, shape)

def salt(img: Image, percentage: int) -> np.ndarray:
    return img.apply_over_channels(salt_channel, percentage)

def salt_channel(channel: np.ndarray, percentage: int) -> np.ndarray:
    p = percentage / 100

    shape = np.shape(channel)
    uniform = rng.rng.uniform(size=channel.size)
    
    noised_channel = channel.flatten()
    noised_channel[uniform < p]   = 0
    noised_channel[uniform > 1-p] = MAX_COLOR

    return np.reshape(noised_channel, shape)
