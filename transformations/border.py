from enum import Enum
from typing import Tuple

import numpy as np
from image import MAX_COLOR, Image

from transformations.utils import index_matrix

from .sliding import PaddingStrategy, sliding_window, weighted_sum

RHO_RESOLUTION = 125
THETA_RESOLUTION = 91
MOST_FITTED_LINES_RATIO = 0.9

class Direction(Enum):
    VERTICAL            = 0
    NEGATIVE_DIAGONAL   = 1
    HORIZONTAL          = 2
    POSITIVE_DIAGONAL   = 3

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, direction: str) -> 'Direction':
        direction_name = direction.upper()
        if direction_name not in Direction.names():
            raise ValueError(f'"{direction_name.title()}" is not a supported direction')
        return cls[direction_name]

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
    # return channel
    return zero_crossing_borders(channel, crossing_threshold)

def susan_channel(channel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = FamousKernel.SUSAN.kernel
    sw = sliding_window(channel, np.shape(kernel), padding_str)
    #Expando dimensiones para que sea compatible con el tamaño de la sliding window
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
