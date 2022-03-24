import functools
from enum import Enum
from typing import Tuple

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

from image_utils import Image


class PaddingStrategy(Enum):
    ZEROS   = functools.partial(lambda c, n: np.pad(c, n, mode='constant', constant_values=0))
    REFLECT = functools.partial(lambda c, n: np.pad(c, n, mode='reflect'))
    EDGE    = functools.partial(lambda c, n: np.pad(c, n, mode='edge'))

    def pad(self, matrix: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return self.value(matrix, (shape[0] - 1) // 2)

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, strategy: str) -> 'PaddingStrategy':
        strategy_name = strategy.upper()
        if strategy_name not in PaddingStrategy.names():
            raise ValueError(f'"{strategy_name.capitalize()}" is not a supported padding strategy')
        return cls[strategy_name]

def is_kernel_valid(kernel: np.ndarray) -> bool:
    return len(kernel.shape) == 2 \
           and kernel.shape[0] == kernel.shape[1] \
           and kernel.shape[0] % 2 == 1 \
           and kernel.shape[0] > 1

def generate_gauss_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 2 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    indices = np.sum(indices**2, axis=2)
    indices = np.exp(-indices / sigma**2)
    return indices / (2 * np.pi * sigma**2)

def sliding_window(matrix: np.ndarray, shape: Tuple[int, ...], padding_str: PaddingStrategy) -> np.ndarray:
    return sliding_window_view(padding_str.pad(matrix, shape), shape)

def weighted_mean(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.sum(sw[:, :] * kernel, axis=(2, 3))
    
def mean_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_mean(channel, np.full((kernel_size, kernel_size), 1 / kernel_size**2), padding_str)

def weighted_median_channel(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.median(np.repeat(sw.reshape(*channel.shape, -1), kernel.flatten(), axis=2), axis=2)

def median_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_median_channel(channel, np.full((kernel_size, kernel_size), 1), padding_str)

def gauss_channel(channel: np.ndarray, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_mean(channel, generate_gauss_kernel(sigma), padding_str)

def high_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), -1 / kernel_size)
    kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2 - 1) / kernel_size
    return weighted_mean(channel, kernel, padding_str)

# ******************* Export Functions ********************** #
def mean(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(mean_channel, kernel_size, padding_str)

def median(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(median_channel, kernel_size, padding_str)

def weighted_median(image: Image, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(weighted_median_channel, kernel, padding_str)

def gauss(image: Image, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(gauss_channel, sigma, padding_str)

def high(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(high_channel, kernel_size, padding_str)
