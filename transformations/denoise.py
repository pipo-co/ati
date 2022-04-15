import functools
from enum import Enum
from typing import Tuple

import numpy as np
from .sliding import PaddingStrategy, sliding_window, sliding_window_tensor, require_valid_kernel, weighted_sum
from image import Image

class DirectionalDerivatives(Enum):
    NORTH = [
        [0,  1,  0],
        [0, -1,  0],
        [0,  0,  0]
    ]
    EAST  = [
        [0,  0,  0],
        [0, -1,  1],
        [0,  0,  0]
    ]
    SOUTH = [
        [0,  0,  0],
        [0, -1,  0],
        [0,  1,  0]
    ]
    WEST = [
        [0,  0,  0],
        [1, -1,  0],
        [0,  0,  0]
    ]

    @classmethod
    def values(cls):
        return list(map(lambda c: np.array(c.value), cls))

    @staticmethod
    def kernel_size() -> Tuple[int, int]:
        return 3, 3

class DiffusionStrategy(Enum):
    ISOTROPIC   = functools.partial(lambda derivatives, sigma: 1)
    LECLERC     = functools.partial(lambda derivatives, sigma: np.exp(-(abs(derivatives) ** 2) / sigma ** 2))
    LORENTZIANO = functools.partial(lambda derivatives, sigma: 1 / ((abs(derivatives) ** 2 / sigma ** 2) + 1))

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.value(*args, **kwargs)

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, strategy: str) -> 'DiffusionStrategy':
        strategy_name = strategy.upper()
        if strategy_name not in DiffusionStrategy.names():
            raise ValueError(f'"{strategy_name.title()}" is not a supported anisotropic function')
        return cls[strategy_name]
    
def mean_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_sum(channel, np.full((kernel_size, kernel_size), 1 / kernel_size**2), padding_str)

def weighted_median_channel(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    require_valid_kernel(kernel)
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.median(np.repeat(sw.reshape(*channel.shape, -1), kernel.flatten(), axis=2), axis=2)

def median_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_median_channel(channel, np.full((kernel_size, kernel_size), 1), padding_str)

def gauss_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 2 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    indices = np.sum(indices**2, axis=2)
    indices = np.exp(-indices / sigma**2)
    return indices / (2 * np.pi * sigma**2)

def gauss_channel(channel: np.ndarray, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_sum(channel, gauss_kernel(sigma), padding_str)

def diffusion_channel(channel: np.ndarray, iterations: int, sigma: int, padding_str: PaddingStrategy, function: DiffusionStrategy) -> np.ndarray:
    new_channel = channel
    for i in range(iterations):
        new_channel = diffusion_step(new_channel, sigma, padding_str, function)
    return new_channel

MAX_ANISOTROPIC_ITERATIONS: int = 20
def diffusion_step(channel: np.ndarray, sigma: int, padding_str: PaddingStrategy, function: DiffusionStrategy) -> np.ndarray:
    sw = sliding_window(channel, DirectionalDerivatives.kernel_size(), padding_str)
    new_channel = channel
    for kernel in DirectionalDerivatives.values():
        derivatives = np.sum(sw[:, :] * kernel, axis=(2, 3))
        new_channel += function(derivatives, sigma) * derivatives / 4
    return new_channel

def bilateral_kernel(sw: np.ndarray, sigma_space: int, sigma_intensity: int, kernel_size: int) -> np.ndarray:
    indexes = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indexes = np.reshape(indexes, (kernel_size, kernel_size, 2))
    spacial_kernel = - np.sum(indexes**2, axis=2) / (2 * sigma_space**2)

    # A cada valor de la ventana se le resta el valor del medio de la ventana, para eso se agregan dos dimensiones
    #  al valor central de la sw para que luego se tenga que estirar contra la coleccion completa
    intensity_kernel = -np.linalg.norm(sw - np.expand_dims(sw[:, :, kernel_size // 2, kernel_size // 2], axis=(2, 3)), axis=4) / (2 * sigma_intensity ** 2)

    result_kernel = np.exp(intensity_kernel - np.expand_dims(spacial_kernel, axis=(0, 1)))
    return result_kernel

def bilateral_filter(image: Image, sigma_space: int, sigma_intensity: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel_size = int(sigma_space * 4 + 1)
    data = image.data

    # Agrego la dimension extra a las imagenes en greyscale para que se comporten como a color
    if image.channels == 1:
        data = np.expand_dims(data, axis=2)

    sw = sliding_window_tensor(data, data[:kernel_size, :kernel_size].shape, padding_str)
    # Mato la dimension que corresponde a la sliding window
    sw = np.squeeze(sw, axis=2)
    kernel = bilateral_kernel(sw, sigma_space, sigma_intensity, kernel_size)
    # Agrego la dimension extra al kernel para que pueda ser multiplicable por el sw
    kernel = np.expand_dims(kernel, axis=4)
    new_data = np.sum(sw * kernel, axis=(2, 3)) / np.sum(kernel, axis=(2, 3))

    return np.squeeze(new_data)

# ******************* Export Functions ********************** #
def mean(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(mean_channel, kernel_size, padding_str)

def median(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(median_channel, kernel_size, padding_str)

def weighted_median(image: Image, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(weighted_median_channel, kernel, padding_str)

def gauss(image: Image, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(gauss_channel, sigma, padding_str)

def diffusion(img: Image, iterations: int, sigma: int, padding_str: PaddingStrategy, function: DiffusionStrategy) -> np.ndarray:
    return img.apply_over_channels(diffusion_channel, iterations, sigma, padding_str, function)

def bilateral(image: Image, sigma_space: int, sigma_intensity: int, padding_str: PaddingStrategy) -> np.ndarray:
    return bilateral_filter(image, sigma_space, sigma_intensity, padding_str)
