import abc
from dataclasses import dataclass
from typing import Tuple

import numpy as np

@dataclass
class DrawCmd(abc.ABC):
    color: Tuple[int, int, int]

    def __init__(self, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        self.color = color

@dataclass
class LineDrawCmd(DrawCmd):
    p1_y: int
    p1_x: int
    p2_y: int
    p2_x: int

    def __init__(self, p1_y: int, p1_x: int, p2_y: int, p2_x: int, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        super().__init__(color)
        self.p1_y = p1_y
        self.p1_x = p1_x
        self.p2_y = p2_y
        self.p2_x = p2_x

@dataclass
class CircleDrawCmd(DrawCmd):
    c_x: int
    c_y: int
    r: float

    def __init__(self, c_x: int, c_y: int, r: float, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        super().__init__(color)
        self.c_x = c_x
        self.c_y = c_y
        self.r = r

@dataclass
class ScatterDrawCmd(DrawCmd):
    points: np.ndarray

    def __init__(self, points: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        super().__init__(color)
        self.points = points
