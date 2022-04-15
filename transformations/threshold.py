import numpy as np

from image import Image, MAX_COLOR, channel_histogram

def channel_threshold(channel: np.ndarray, t: int) -> np.ndarray:
    ret = np.zeros(channel.shape)
    ret[channel > t] = MAX_COLOR
    return ret

def channel_global(channel: np.ndarray, t: int) -> np.ndarray:
    old_t = 0
    new_t = t
    while abs(old_t - new_t) > 1:
        old_t = new_t
        minor_umbral = np.mean(channel[channel < old_t])
        mayor_umbral = np.mean(channel[channel >= old_t])
        new_t = (minor_umbral + mayor_umbral) // 2

    print(f'Global Umbral: {new_t}')
    return channel_threshold(channel, new_t)

# intra_variance = (p1.m0 - p0.m1)^2 / p0.p1
def channel_otsu(channel: np.ndarray) -> np.ndarray:
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

    return channel_threshold(channel, t)

# ******************* Export Functions ********************** #

def manual(img: Image, t: int) -> np.ndarray:
    return img.apply_over_channels(channel_threshold, t)

def global_(img: Image, umb: int) -> np.ndarray:
    return img.apply_over_channels(channel_global, umb)

def otsu(image: Image) -> np.ndarray:
    return image.apply_over_channels(channel_otsu)
