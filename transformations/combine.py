from typing import Tuple, List
import numpy as np
import cv2

from models.image import Image, ImageChannelTransformation, normalize

def add(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.add(first_img.data, second_img.data), []

def sub(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.subtract(first_img.data, second_img.data), []

def multiply(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.multiply(first_img.data, second_img.data), []

# Por ahora asumimos gris
def sift(img1: Image, img2: Image, features: int, layers: int, contrast_t: float, edge_t: float, sigma: float, match_t: float, cross_check: bool) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    new_data: np.ndarray
    channels_tr: List[ImageChannelTransformation] = []

    if img1.channels == 1:
        fn_ret = sift_channel(img1.data, img2.data, features=features, layers=layers, contrast_t=contrast_t, edge_t=edge_t, sigma=sigma, match_t=match_t, cross_check=cross_check, color=(0, 0, 255))
        if isinstance(fn_ret, tuple):
            new_data = fn_ret[0]
            channels_tr.append(fn_ret[1])
        else:
            new_data = fn_ret
    else:
        new_data = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3))
        for channel in range(img1.channels):
            fn_ret = sift_channel(img1.get_channel(channel), img2.get_channel(channel), features=features, layers=layers, contrast_t=contrast_t, edge_t=edge_t, sigma=sigma, match_t=match_t, cross_check=cross_check)
            if isinstance(fn_ret, tuple):
                new_data[:, :, channel] = fn_ret[0][:,:,0]
                channels_tr.append(fn_ret[1])
            else:
                new_data[:, :, channel] = fn_ret[:,:,0]

    return new_data, channels_tr

def sift_channel(channel1: np.ndarray, channel2: np.ndarray, features: int, layers: int, contrast_t: float, edge_t: float, sigma: float, match_t: float, cross_check: bool, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    data1 = normalize(channel1)
    data2 = normalize(channel2)

    sift_handler = cv2.SIFT_create(nfeatures=features, nOctaveLayers=layers, contrastThreshold=contrast_t, edgeThreshold=edge_t, sigma=sigma)

    kp1, desc1 = sift_handler.detectAndCompute(data1, None)
    kp2, desc2 = sift_handler.detectAndCompute(data2, None)

    matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=cross_check)

    matches = matcher.radiusMatch(desc1, desc2, match_t)
    matches = tuple([t[0] for t in matches if t])  # Flateneo

    match_ratio = len(matches) / len(kp1)

    data1 = cv2.drawKeypoints(data1, kp1, None, color=(255 - color[0], 255 - color[1], 255 - color[2]), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    new_data = cv2.drawMatches(data1, kp1, data2, kp2, matches, None, matchColor=color, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    result = ImageChannelTransformation({'Match Ratio': round(match_ratio, 2)}, {})
    return new_data, result
