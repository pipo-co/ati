from enum import Enum

import numpy as np

from image import Image, MAX_COLOR
from .sliding import PaddingStrategy, weighted_sum

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

    # TODO(faus, nacho): link a stack overflow
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
        [ 1,  1,  1],
        [ 1, -2,  1],
        [-1, -1, -1]
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
    kernel_size = int(sigma * 2 + 1)
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
