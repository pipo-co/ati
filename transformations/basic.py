from typing import Tuple
import numpy as np

from models.image import Image, MAX_COLOR, ImageChannelTransformation, ImageTransformation, normalize, channel_histogram

def channel_equalization(channel: np.ndarray) -> Tuple[np.ndarray, ImageChannelTransformation]:
    channel = normalize(channel, np.int64)
    normed_hist, bins = channel_histogram(channel)
    s = normed_hist.cumsum()
    masked_s = np.ma.masked_equal(s, 0)
    masked_min = masked_s.min()
    masked_s = (masked_s - masked_min) * MAX_COLOR / (masked_s.max() - masked_min)
    s = np.ma.filled(masked_s, 0)
    return s[channel], ImageChannelTransformation() 

# ******************* Export Functions ********************** #

def power(img: Image, gamma: float) -> Tuple[np.ndarray, ImageTransformation]:
    c = MAX_COLOR**(1 - gamma)
    return c*(img.data**gamma), ImageTransformation('power', gamma=gamma)

def negate(img: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return np.array([MAX_COLOR - xi for xi in img.data]), ImageTransformation('negate')

def equalize(image: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('equalize', channel_equalization)
