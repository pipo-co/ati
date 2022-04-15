from enum import Enum
from typing import List

import numpy as np
from sliding import PaddingStrategy, sliding_window, sliding_window_tensor
from image_utils import Image, channel_histogram, channel_to_binary, MAX_COLOR


class DirectionalOperator(Enum):
    VERTICAL            = 0
    NEGATIVE_DIAGONAL   = 1
    HORIZONTAL          = 2
    POSITIVE_DIAGONAL   = 3

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_str(cls, direction: str) -> 'DirectionalOperator':
        direction_name = direction.upper()
        if direction_name not in DirectionalOperator.names():
            raise ValueError(f'"{direction_name.title()}" is not a supported direction')
        return cls[direction_name]

def is_kernel_valid(kernel: np.ndarray) -> bool:
    return len(kernel.shape) == 2 \
           and kernel.shape[0] == kernel.shape[1] \
           and kernel.shape[0] % 2 == 1 \
           and kernel.shape[0] > 1

def generate_gauss_kernel(sigma: float) -> np.ndarray:
    kernel_size = int(sigma * 2 + 1)
    indices = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indices = np.reshape(indices, (kernel_size, kernel_size, 2))
    indices = np.sum(indices**2, axis=2)
    indices = np.exp(-indices / sigma**2)
    return indices / (2 * np.pi * sigma**2)

def weighted_sum(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.sum(sw[:, :] * kernel, axis=(2, 3))
    
def mean_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_sum(channel, np.full((kernel_size, kernel_size), 1 / kernel_size**2), padding_str)

def weighted_median_channel(channel: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    if not is_kernel_valid(kernel):
        raise ValueError(f'Invalid kernel: {kernel}')
    sw = sliding_window(channel, kernel.shape, padding_str)
    return np.median(np.repeat(sw.reshape(*channel.shape, -1), kernel.flatten(), axis=2), axis=2)

def median_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_median_channel(channel, np.full((kernel_size, kernel_size), 1), padding_str)

def gauss_channel(channel: np.ndarray, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return weighted_sum(channel, generate_gauss_kernel(sigma), padding_str)

def high_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = np.full((kernel_size, kernel_size), -1 / kernel_size)
    kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2 - 1) / kernel_size
    return weighted_sum(channel, kernel, padding_str)

def modulus_derivative_synthesis(img: np.ndarray, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    y_channel = weighted_sum(img, kernel, padding_str)
    kernel = np.rot90(kernel)
    x_channel = weighted_sum(img, kernel, padding_str)
    return np.sqrt(y_channel ** 2 + x_channel ** 2)

def prewitt_kernel(kernel_size: int) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[0, :] = -1
    kernel[-1, :] = 1
    return kernel

def prewitt_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return modulus_derivative_synthesis(channel, prewitt_kernel(kernel_size), padding_str)

def sobel_kernel(kernel_size: int) -> np.ndarray:
    kernel = prewitt_kernel(kernel_size)
    kernel[0, kernel_size//2] = -2
    kernel[-1, kernel_size//2] = 2
    return kernel

def sobel_channel(channel: np.ndarray, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return modulus_derivative_synthesis(channel, sobel_kernel(kernel_size), padding_str)


STANDARD_DERIVATIVE_KERNEL: List[List[int]] = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

ALTERNATIVE_DERIVATIVE_KERNEL: List[List[int]] = [
    [ 1,  1,  1],
    [ 1, -2,  1],
    [-1, -1, -1]
]

# intra_variance = (p1.m0 - p0.m1)^2 / p0.p1
def channel_otsu_threshold(channel: np.ndarray) -> np.ndarray:
    hist, bins = channel_histogram(channel)
    hist = hist[:-1]

    p = np.cumsum(hist)
    m = np.cumsum(np.arange(hist.size) * hist)
    mg = m[-1]

    intra_variance = (mg*p - m)**2 / (p * (1-p))
    max_variance = np.ravel(np.where(intra_variance == np.amax(intra_variance)))
    t = int(max_variance.mean().round())
    print(f'Otsu Umbral Chosen: {t}')

    return channel_to_binary(channel, t)

def directional_channel(channel: np.ndarray, base_kernel: List[List[int]], rotations: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel = np.array(base_kernel)
    kernel = rotate_steps(kernel, rotations)
    return weighted_sum(channel, kernel, padding_str)

def outer_slice(x):
    return np.r_[x[0], x[1:-1, -1], x[-1, :0:-1], x[-1:0:-1, 0]]

def rotate_steps(x, shift):
    out = np.empty_like(x)
    N = x.shape[0]
    idx = np.arange(x.size).reshape(x.shape)
    for n in range((N+1) // 2):
        sliced_idx = outer_slice(idx[n:N-n, n:N-n])
        out.ravel()[sliced_idx] = np.roll(np.take(x, sliced_idx), shift)
    return out

# ******************* Export Functions ********************** #
def mean(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(mean_channel, kernel_size, padding_str)

def median(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(median_channel, kernel_size, padding_str)

def weighted_median(image: Image, kernel: np.ndarray, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(weighted_median_channel, kernel, padding_str)

def gauss(image: Image, sigma: float, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(gauss_channel, sigma, padding_str)

def high(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(high_channel, kernel_size, padding_str)

def directional(image: Image, kernel: List[List[int]], padding_str: PaddingStrategy, rotations: int) -> np.ndarray:
    return image.apply_over_channels(directional_channel, kernel, rotations, padding_str)

def prewitt(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(prewitt_channel, kernel_size, padding_str)

def sobel(image: Image, kernel_size: int, padding_str: PaddingStrategy) -> np.ndarray:
    return image.apply_over_channels(sobel_channel, kernel_size, padding_str)

def otsu_threshold(image: Image) -> np.ndarray:
    return image.apply_over_channels(channel_otsu_threshold)

def bilateral_filter(image: Image, sigma_space: int, sigma_intensity: int, padding_str: PaddingStrategy) -> np.ndarray:
    kernel_size = int(sigma_space * 4 + 1)

    data = image.data

    #Agrego la dimension extra a las imagenes en greyscale para que se comporten como a color
    if image.channels == 1:
        data = np.expand_dims(data, axis=2)

    sw = sliding_window_tensor(data, data[:kernel_size, :kernel_size].shape, padding_str)
    #Mato la dimension que corresponde a la sliding window
    sw = np.squeeze(sw, axis=2)
    kernel = generate_bilateral_kernel(sw, sigma_space, sigma_intensity, kernel_size)
    #Agrego la dimension extra al kernel para que pueda ser multiplicable por el sw
    kernel = np.expand_dims(kernel, axis=4)
    new_data = np.sum(sw * kernel, axis=(2,3)) / np.sum(kernel, axis=(2,3))

    return np.squeeze(new_data)

def generate_bilateral_kernel(sliding_window: np.ndarray, sigma_space: int, sigma_intensity: int, kernel_size: int) -> np.ndarray:

    indexes = np.array(list(np.ndindex((kernel_size, kernel_size)))) - kernel_size//2 # noqa
    indexes = np.reshape(indexes, (kernel_size, kernel_size, 2))
    spacial_kernel = - np.sum(indexes**2, axis=2) / (2* sigma_space**2)

    #A cada valor de la ventana se le resta el valor del medio de la ventana, para eso se agregan dos dimensiones
    #al valor central de la sw para que luego se tenga que estirar contra la coleccion completa
    intensity_kernel = -np.linalg.norm(sliding_window - np.expand_dims(sliding_window[:,:,kernel_size // 2, kernel_size//2], axis=(2,3)), axis=4) / (2*sigma_intensity**2)
    result_kernel = np.exp(intensity_kernel - np.expand_dims(spacial_kernel, axis=(0,1)))

    return result_kernel