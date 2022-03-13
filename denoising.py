import functools
from enum import Enum
from typing import Tuple

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class PaddingStrategy(Enum):
    ZEROS   = functools.partial(lambda c, n: np.pad(c, n, mode='constant', constant_values=0))
    REFLECT = functools.partial(lambda c, n: np.pad(c, n, mode='reflect'))
    EDGE    = functools.partial(lambda c, n: np.pad(c, n, mode='edge'))

    def pad(self, matrix: np.ndarray, shape: np.ndarray) -> np.ndarray:
        return self.value(matrix, (shape[0] - 1) // 2)

    @classmethod
    def values(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, strategy: str) -> 'PaddingStrategy':
        strategy_name = strategy.upper()

        if strategy_name not in PaddingStrategy.values():
            raise ValueError(f'"{strategy_name}" is not a supported padding strategy')
        return cls[strategy_name]


def is_kernel_valid(kernel: np.ndarray) -> bool:
    return len(kernel.shape) == 2 \
           and kernel.shape[0] == kernel.shape[1] \
           and kernel.shape[0] % 2 == 1 \
           and kernel.shape[0] > 1


def sliding_window(matrix: np.ndarray, shape: Tuple[int], padding_strategy: str) -> np.ndarray:
    padded = PaddingStrategy.from_str(padding_strategy).pad(matrix, shape)
    return sliding_window_view(padded, shape)


def weighted_mean(channel: np.ndarray, kernel: np.ndarray, padding_strategy: str) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')

    sw = sliding_window(channel, kernel.shape, padding_strategy)
    return np.mean(sw[:, :] * kernel, axis=(2, 3)).astype(np.uint8)


def mean(channel: np.ndarray, n: int, padding_strategy: str) -> np.ndarray:
    return weighted_mean(channel, np.full((n, n), 1), padding_strategy)


def weighted_median(channel: np.ndarray, kernel: np.ndarray, padding_strategy: str) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')

    sw = sliding_window(channel, kernel.shape, padding_strategy)
    return np.median(sw[:, :] * kernel, axis=(2, 3)).astype(np.uint8)


def median(channel: np.ndarray, n: int, padding_strategy: str) -> np.ndarray:
    return weighted_median(channel, np.full((n, n), 1), padding_strategy)
