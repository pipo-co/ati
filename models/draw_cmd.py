from ast import Tuple
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class LineDrawCmd:
    p1_y: int
    p1_x: int
    p2_y: int
    p2_x: int


@dataclass
class CircleDrawCmd:
    c_x: int
    c_y: int
    r: float

@dataclass
class ScatterDrawCmd:
    points: np.ndarray

DrawCmd = Union[LineDrawCmd, CircleDrawCmd, ScatterDrawCmd]

