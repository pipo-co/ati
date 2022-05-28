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
    data1 = normalize(img1.data)
    data2 = normalize(img2.data)

    sift_handler = cv2.SIFT_create(nfeatures=features, nOctaveLayers=layers, contrastThreshold=contrast_t, edgeThreshold=edge_t, sigma=sigma)

    kp1, desc1 = sift_handler.detectAndCompute(data1, None)
    kp2, desc2 = sift_handler.detectAndCompute(data2, None)

    matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=cross_check)

    matches = matcher.radiusMatch(desc1, desc2, match_t)
    matches = tuple([t[0] for t in matches if t])  # Flateneo

    match_ratio = len(matches) / len(kp1)

    data1 = cv2.drawKeypoints(data1, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    new_data = cv2.drawMatches(data1, kp1, data2, kp2, matches, None, matchColor=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    result = ImageChannelTransformation({'Match Ratio': round(match_ratio, 2)}, {})
    return new_data, [result]
