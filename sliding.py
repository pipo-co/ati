from enum import Enum
import functools
from typing import Tuple

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


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

def sliding_window(matrix: np.ndarray, shape: Tuple[int, ...], padding_str: PaddingStrategy) -> np.ndarray:
    return sliding_window_view(padding_str.pad(matrix, shape), shape)