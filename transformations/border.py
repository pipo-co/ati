from enum import Enum
from typing import Tuple

import numpy as np
from models.image import MAX_COLOR, Image, normalize

from transformations.utils import index_matrix
from .sliding import PaddingStrategy, sliding_window, weighted_sum

RHO_RESOLUTION = 125
THETA_RESOLUTION = 91
MOST_FITTED_LINES_RATIO = 0.9

class Direction(Enum):
    VERTICAL = [
        [ 0,  1,  0],
        [ 0,  1,  0],
        [ 0,  1,  0]
    ]
    NEGATIVE_DIAGONAL = [
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ]
    HORIZONTAL = [
        [ 0,  0,  0],
        [ 1,  1,  1],
        [ 0,  0,  0]
    ]
    POSITIVE_DIAGONAL = [
        [ 0,  0,  1],
        [ 0,  1,  0],
        [ 1,  0,  0]
    ]

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, direction: str) -> 'Direction':
        direction_name = direction.upper()
        if direction_name not in Direction.names():
            raise ValueError(f'"{direction_name.title()}" is not a supported direction')
        return cls[direction_name]

    @classmethod
    def from_angle(cls, angle: int) -> 'Direction':
        if angle == 0:
            return cls.HORIZONTAL
        elif angle == 45:
            return cls.POSITIVE_DIAGONAL
        elif angle == 90:
            return cls.VERTICAL
        elif angle == 135:
            return cls.NEGATIVE_DIAGONAL
        else:
            raise ValueError(f'{angle} is not a valid direction angle')

    @property
    def kernel(self) -> np.ndarray:
        return np.array(self.value)

    # https://stackoverflow.com/a/41506120/12270520
    @staticmethod
    def _outer_slice(x):
        return np.r_[x[0], x[1:-1, -1], x[-1, :0:-1], x[-1:0:-1, 0]]

    @staticmethod
    def _rotate_matrix(x, shift):
        out = np.empty_like(x)
        N = x.shape[0]
        idx = np.arange(x.size).reshape(x.shape)
        for n in range((N + 1) // 2):
            sliced_idx = Direction._outer_slice(idx[n:N - n, n:N - n])
            out.ravel()[sliced_idx] = np.roll(np.take(x, sliced_idx), shift)
        return out

    def align_vertical_kernel(self, kernel: np.ndarray):
        return self._rotate_matrix(kernel, self.value)

class FamousKernel(Enum):
    # x derivative kernel
    PREWITT = [
        [ 1,  0, -1],
        [ 1,  0, -1],
        [ 1,  0, -1]
    ]
    SOBEL = [
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ]
    LAPLACE = [
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]
    # TODO(tobi): Se cambio el kernel de juliana?? Igual yo no importa
    JULIANA = [
        [ 1,  1, -1],
        [ 1, -2, -1],
        [ 1,  1, -1]
    ]
    SUSAN = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ]

    @property
    def kernel(self) -> np.ndarray:
        return np.array(self.value)

def directional_channel(channel: np.ndarray, vertical_kernel: np.ndarray, border_dir: Direction, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = border_dir.align_vertical_kernel(vertical_kernel)
    return weighted_sum(channel, kernel, padding_str)

def high_pass_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), -1 / kernel_size)
    kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2 - 1) / kernel_size
    return weighted_sum(channel, kernel, padding_str)

def gradient_modulus(img: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    y_channel = weighted_sum(img, kernel, padding_str)
    kernel = np.rot90(kernel)
    x_channel = weighted_sum(img, kernel, padding_str)
    return np.sqrt(y_channel ** 2 + x_channel ** 2)

def prewitt_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return gradient_modulus(channel, FamousKernel.PREWITT.kernel, padding_str)

def sobel_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return gradient_modulus(channel, FamousKernel.SOBEL.kernel, padding_str)

def zero_crossing_vertical(data: np.ndarray, threshold: int) -> np.ndarray:
    ans = np.empty(data.shape, dtype=np.bool8)
    # Cambios de signo directos
    ans[:-1] = (data[:-1] * data[1:] < 0) & (np.abs(data[:-1] - data[1:]) > threshold)
    # Cambios con un 0 en el medio
    ans[:-2] |= (data[:-2] * data[2:] < 0) & (data[1:-1] == 0) & (np.abs(data[:-2] - data[2:]) > threshold)
    # Ultimo nunca cruza
    ans[-1] = False

    return ans

def zero_crossing_horizontal(data: np.ndarray, threshold: int) -> np.ndarray:
    ans = np.empty(data.shape, dtype=np.bool8)
    # Cambios de signo directos
    ans[:, :-1] = (data[:, :-1] * data[:, 1:] < 0) & (np.abs(data[:, :-1] - data[:, 1:]) > threshold)
    # Cambios con un 0 en el medio
    ans[:, :-2] |= (data[:, :-2] * data[:, 2:] < 0) & (data[:, 1:-1] == 0) & (np.abs(data[:, :-2] - data[:, 2:]) > threshold)
    # Ultimo nunca cruza
    ans[:, -1] = False

    return ans

def zero_crossing_borders(data: np.ndarray, threshold: int) -> np.ndarray:
    mask = zero_crossing_vertical(data, threshold) | zero_crossing_horizontal(data, threshold)
    ret = np.zeros(data.shape)
    ret[mask] = MAX_COLOR
    return ret

def laplacian_channel(channel: np.ndarray, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    # Derivada segunda
    channel = weighted_sum(channel, FamousKernel.LAPLACE.kernel, padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold)

def log_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 10 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    sum_squared_over_sigma = np.sum(indices**2, axis=2) / sigma**2  # (x^2 + y^2) / sigma^2
    k = (np.sqrt(2 * np.pi) * sigma**3)                             # sqrt(2pi) * sigma^3
    return - ((2 - sum_squared_over_sigma) / k) * np.exp(-sum_squared_over_sigma/2)

def log_channel(channel: np.ndarray, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    # Derivada segunda con gauss
    channel = weighted_sum(channel, log_kernel(sigma), padding_str)

    # Queremos ver donde se hace 0, pues son los minimos/maximos de la derivada => borde
    return zero_crossing_borders(channel, crossing_threshold)

def susan_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = FamousKernel.SUSAN.kernel
    sw = sliding_window(channel, np.shape(kernel), padding_str)
    # Expando dimensiones para que sea compatible con el tamaño de la sliding window
    new_channel = np.expand_dims(channel, axis=(2,3))
    absolute_values = np.absolute(sw[:,:]*kernel - new_channel[:,:])
    absolute_values[absolute_values < 15] = 1
    absolute_values[absolute_values >= 15] = 0
    values = 1 - np.sum(absolute_values, axis=(2, 3)) / kernel.size
    values[(values < 0.4) | (values >= 0.85)] = 0
    values[(values >= 0.4) & (values < 0.65)] = 63
    values[(values >= 0.65) & (values < 0.85)] = 255
    return values

def hough_channel(channel: np.ndarray, t: float) -> np.ndarray:
    p     = np.sqrt(2) * np.max(channel.shape)
    theta = np.linspace(-np.pi/2, np.pi/2, THETA_RESOLUTION)
    rho   = np.linspace(-p, p, RHO_RESOLUTION)

    indices = np.insert(index_matrix(*channel.shape), 0, -1, axis=2)
    indices = np.expand_dims(indices, axis=(0, 1))

    acum = np.stack(np.meshgrid(rho, theta), -1)
    acum = np.dstack((acum, acum[:,:,1]))
    acum[:,:,1] = np.sin(acum[:,:,1])
    acum[:,:,2] = np.cos(acum[:,:,2])
    acum = np.expand_dims(acum, axis=(2, 3))

    # |rho - x*cos(theta) - y*sin(theta)|
    lines = (np.abs(np.sum(acum * indices, axis=4))) < t
    white_points = channel > 0

    points_in_line = (white_points & lines).sum(axis=(2,3))
    most_fitted_lines = np.argwhere(points_in_line > MOST_FITTED_LINES_RATIO * points_in_line.max())
    print(np.take(acum, most_fitted_lines))
    # TODO(nacho): Dibujar linea

def canny_drag_borders(gradient_mod: np.ndarray, t1: int, t2: int, max_col: int, max_row: int, row: int, col: int) -> None:
    if t1 < gradient_mod[row, col] < t2:
        # Conectado por un borde de manera 4-conexo
        touches_border = False
        if not touches_border and row - 1 >= 0:
            touches_border = gradient_mod[row - 1, col] == MAX_COLOR
        if not touches_border and col - 1 >= 0:
            touches_border = gradient_mod[row, col - 1] == MAX_COLOR
        if not touches_border and row + 1 < max_row:
            touches_border = gradient_mod[row + 1, col] == MAX_COLOR
        if not touches_border and col + 1 < max_col:
            touches_border = gradient_mod[row, col + 1] == MAX_COLOR

        gradient_mod[row, col] = MAX_COLOR if touches_border else 0

# Asume que ya fue paso por un filtro gaussiano
def canny_channel(channel: np.ndarray, t1: int, t2: int, padding_str: PaddingStrategy) -> np.ndarray:
    # Usamos prewitt para derivar
    kernel = FamousKernel.PREWITT.kernel
    dy = weighted_sum(channel, kernel, padding_str)
    kernel = np.rot90(kernel)
    dx = weighted_sum(channel, kernel, padding_str)
    # TODO(tobi): modulo o abs?
    # gradient_mod = np.sqrt(dy ** 2 + dx ** 2)
    gradient_mod = np.abs(dy) + np.abs(dx)

    # Calculamos el angulo de la derivada en grados
    d_angle = np.arctan2(dy, dx)
    d_angle[d_angle < 0] += np.pi
    d_angle = np.rad2deg(d_angle)

    # Discretizamos dicho angulo
    dir_sw = np.empty((*d_angle.shape, *kernel.shape))
    dir_sw[((0 <= d_angle) & (d_angle < 22.5)) | ((157.5 <= d_angle) & (d_angle <= 180))]    = Direction.from_angle(0).kernel
    dir_sw[(22.5 <= d_angle) & (d_angle < 67.5)]                                            = Direction.from_angle(45).kernel
    dir_sw[(67.5 <= d_angle) & (d_angle < 112.5)]                                           = Direction.from_angle(90).kernel
    dir_sw[(112.5 <= d_angle) & (d_angle < 157.5)]                                          = Direction.from_angle(135).kernel

    # Suprimimos los valores que no son maximos
    max_suppression_sw = sliding_window(gradient_mod, kernel.shape, padding_str) * dir_sw
    max_suppression_sw = np.max(max_suppression_sw, axis=(2, 3))
    gradient_mod[max_suppression_sw != gradient_mod] = 0

    # Normalizamos la imagen antes del thresholding
    gradient_mod = normalize(gradient_mod, np.uint64)

    # Thresholding con histéresis
    gradient_mod[gradient_mod >= t2] = MAX_COLOR
    gradient_mod[gradient_mod <= t1] = 0

    max_row, max_col = gradient_mod.shape

    # Arrastramos los bordes de manera vertical
    for row in range(max_row):
        for col in range(max_col):
            canny_drag_borders(gradient_mod, t1, t2, max_col, max_row, row, col)

    # Arrastramos los bordes de manera horizontal
    for col in range(max_col):
        for row in range(max_row):
            canny_drag_borders(gradient_mod, t1, t2, max_col, max_row, row, col)

    return gradient_mod

# ******************* Export Functions ********************** #
def directional(image: Image, kernel: FamousKernel, border_dir: Direction, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(directional_channel, kernel.kernel, border_dir, padding_str)

def high_pass(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(high_pass_channel, kernel_size, padding_str)

def prewitt(image: Image, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(prewitt_channel, padding_str)

def sobel(image: Image, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(sobel_channel, padding_str)

def laplace(image: Image, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(laplacian_channel, crossing_threshold, padding_str)

def log(image: Image, sigma: float, crossing_threshold: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(log_channel, sigma, crossing_threshold, padding_str)

def susan(image: Image, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(susan_channel, padding_str)

def hough(image: Image, t: float) -> np.ndarray:
    return image.apply_over_channels(hough_channel, t)

def canny(image: Image, t1: int, t2: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(canny_channel, t1, t2, padding_str)

def active_outline_first_frame(image: Image, p1: Tuple[int, int], p2: Tuple[int, int]) -> np.ndarray:
    pass

def active_outline_middle_frame(image: Image, in_color: int, l_in: np.ndarray, l_out: np.ndarray) -> np.ndarray:
    pass
