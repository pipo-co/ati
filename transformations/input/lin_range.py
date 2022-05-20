from dataclasses import dataclass

import numpy as np


@dataclass
class LinRange:
    start:  float
    end:    float
    count:  int

    def __init__(self, start: float, end: float, count: int) -> None:
        if count < 2:
            raise ValueError('Count must be at least 2')

        self.start  = start
        self.end    = end
        self.count  = count

    def to_linspace(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.count)
