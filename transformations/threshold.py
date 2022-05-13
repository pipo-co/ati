from typing import Tuple
import numpy as np

from models.image import Image, MAX_COLOR, ImageChannelTransformation, ImageTransformation, channel_histogram

def binary_threshold(channel: np.ndarray, threshold:int) -> np.ndarray:
    ret = np.zeros(channel.shape)
    ret[channel > threshold] = MAX_COLOR
    return ret

def channel_threshold(channel: np.ndarray, threshold: int) -> Tuple[np.ndarray, ImageChannelTransformation]:
    return binary_threshold(channel, threshold), ImageChannelTransformation()

def channel_global(channel: np.ndarray, threshold: int) -> Tuple[np.ndarray, ImageChannelTransformation]:
    old_t = 0
    new_t = threshold
    while abs(old_t - new_t) > 1:
        old_t = new_t
        minor_umbral = np.mean(channel[channel < old_t])
        mayor_umbral = np.mean(channel[channel >= old_t])
        new_t = (minor_umbral + mayor_umbral) // 2

    print(f'Global Umbral: {new_t}')
    return binary_threshold(channel, new_t), ImageChannelTransformation(new_t=new_t)

# intra_variance = (p1.m0 - p0.m1)^2 / p0.p1
def channel_otsu(channel: np.ndarray) -> Tuple[np.ndarray, ImageChannelTransformation]:
    hist, bins = channel_histogram(channel)

    p = np.cumsum(hist)
    m = np.cumsum(np.arange(hist.size) * hist)
    mg = m[-1]

    # intra_variance(L - 1) = undefined => La matamos
    p = p[:-1]
    m = m[:-1]

    intra_variance = (mg*p - m)**2 / (p * (1-p))
    max_variance = np.ravel(np.where(intra_variance == np.amax(intra_variance)))
    t = int(max_variance.mean().round())
    print(f'Otsu Umbral Chosen: {t}')

    return binary_threshold(channel, t), ImageChannelTransformation(new_t=t)

# ******************* Export Functions ********************** #

def manual(img: Image, threshold: int) -> Tuple[np.ndarray, ImageTransformation]:
    return img.apply_over_channels('threshold', channel_threshold, threshold=threshold)

def global_(img: Image, threshold: int) -> Tuple[np.ndarray, ImageTransformation]:
    return img.apply_over_channels('global', channel_global, threshold=threshold)

def otsu(image: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return image.apply_over_channels('global', channel_otsu)
