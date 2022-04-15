from enum import Enum
import functools
from typing import Tuple

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

class PaddingStrategy(Enum):
    REFLECT = functools.partial(lambda c, n: np.pad(c, n, mode='reflect'))
    EDGE    = functools.partial(lambda c, n: np.pad(c, n, mode='edge'))
    ZEROS   = functools.partial(lambda c, n: np.pad(c, n, mode='constant', constant_values=0))

    def pad(self, matrix: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return self.value(matrix, (shape[0] - 1) // 2)

    def pad_tensor(self, tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        pad_size = (shape[0] - 1) // 2
        return self.value(tensor, ((pad_size, pad_size), (pad_size, pad_size), (0,0)))

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, strategy: str) -> 'PaddingStrategy':
        strategy_name = strategy.upper()
        if strategy_name not in PaddingStrategy.names():
            raise ValueError(f'"{strategy_name.title()}" is not a supported padding strategy')
        return cls[strategy_name]

def is_kernel_valid(kernel: np.ndarray) -> bool:
    return len(kernel.shape) == 2 \
           and kernel.shape[0] == kernel.shape[1] \
           and kernel.shape[0] % 2 == 1 \
           and kernel.shape[0] > 1

def require_valid_kernel(kernel: np.ndarray) -> None:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel.shape}')

def sliding_window(matrix: np.ndarray, shape: Tuple[int, ...], padding_str: PaddingStrategy) -> np.ndarray:
    return sliding_window_view(padding_str.pad(matrix, shape), shape)

def sliding_window_tensor(matrix: np.ndarray, shape: Tuple[int, ...], padding_str: PaddingStrategy) -> np.ndarray:
    return sliding_window_view(padding_str.pad_tensor(matrix, shape), shape)

def weighted_sum(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    require_valid_kernel(kernel)
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.sum(sw[:, :] * kernel, axis=(2, 3))
