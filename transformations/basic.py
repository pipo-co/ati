import numpy as np

from image import Image, MAX_COLOR, normalize, channel_histogram

def channel_equalization(channel: np.ndarray)  -> np.ndarray:
    channel = normalize(channel, np.int64)
    normed_hist, bins = channel_histogram(channel)
    s = normed_hist.cumsum()
    masked_s = np.ma.masked_equal(s, 0)
    masked_min = masked_s.min()
    masked_s = (masked_s - masked_min) * MAX_COLOR / (masked_s.max() - masked_min)
    s = np.ma.filled(masked_s, 0)
    return s[channel]

# ******************* Export Functions ********************** #

def power(img: Image, gamma: float) -> np.ndarray:
    c = MAX_COLOR**(1 - gamma)
    return c*(img.data**gamma)

def negate(img: Image) -> np.ndarray:
    return np.array([MAX_COLOR - xi for xi in img.data])

def equalize(image: Image) -> np.ndarray:
    return image.apply_over_channels(channel_equalization)
