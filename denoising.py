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

def sliding_window(matrix: np.ndarray, shape: Tuple[int, ...], padding_str: PaddingStrategy) -> np.ndarray:
    return sliding_window_view(padding_str.pad(matrix, shape), shape)

def weighted_mean(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.mean(sw[:, :] * kernel, axis=(2, 3)).astype(np.uint8)

def mean_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_mean(channel, np.full((kernel_size, kernel_size), 1), padding_str)

def weighted_median_channel(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.median(np.repeat(sw.reshape(*channel.shape, -1), kernel.flatten(), axis=2), axis=2).astype(np.uint8)

def median_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_median_channel(channel, np.full((kernel_size, kernel_size), 1), padding_str)

# ******************* Export Functions ********************** #
def mean(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(mean_channel, kernel_size, padding_str)

def median(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(median_channel, kernel_size, padding_str)

def weighted_median(image: Image, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(weighted_median_channel, kernel, padding_str)
