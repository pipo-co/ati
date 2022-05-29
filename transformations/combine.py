from typing import Tuple, List
import numpy as np
import cv2

from models.image import Image, ImageChannelTransformation, normalize
from transformations.data_models import Measurement


def sift_channel(channel1: np.ndarray, channel2: np.ndarray, features: int, layers: int, contrast_t: float,
                 edge_t: float, sigma: float, match_t: float, cross_check: bool, multi_channel: bool,
                 color: Tuple[int, int, int] = (255, 255, 255)
                 ) -> Tuple[np.ndarray, ImageChannelTransformation]:
    if multi_channel and not all(elem == color[0] for elem in color):
        raise ValueError(f'If multi channel, color must be a shade of gray. This means all elements of tuple must be the same value. Not true for {color}')

    # Normalizamos porque OpenCV solo se banca uint8
    data1 = normalize(channel1)
    data2 = normalize(channel2)

    # Parametros propios del metodo:
    # features:     Cantidad de keypoints a tomar
    # layers:       Cantidad de capas que se generan a la hora de buscar keypoints
    # contrast_t:   Threshold que debe superar para considerar un keypoint. CUanto as alto menos keypoints. Se divide por la cantidad de layers.
    # edge_t:       Threshold que se usa para determinar "edge-like features" (?). Cuanto mas alto, mas keypoints hay.
    # sigma:        Sigma usado en los gauss
    sift_handler = cv2.SIFT_create(nfeatures=features, nOctaveLayers=layers, contrastThreshold=contrast_t, edgeThreshold=edge_t, sigma=sigma)

    kp1, desc1 = sift_handler.detectAndCompute(data1, None)
    kp2, desc2 = sift_handler.detectAndCompute(data2, None)

    # Emparejamos los descriptores a distancia L2 con un maximo de match_t
    matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=cross_check)  # Cross check = tiene que emparejarse de ambos lados
    matches = matcher.radiusMatch(desc1, desc2, match_t)
    matches = tuple([t[0] for t in matches if t])           # Flateneamos
    matches = sorted(matches, key=lambda x: x.distance)     # Ordenamos de mejor a peor

    if not multi_channel:
        # Solo mostramos los keypoints no encontrados en imagenes grises (color inverso al seleccionado)
        data1 = cv2.drawKeypoints(data1, kp1, None, color=(255 - color[0], 255 - color[1], 255 - color[2]), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    new_data = cv2.drawMatches(data1, kp1, data2, kp2, matches, None, matchColor=color, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    if multi_channel:
        # Sacamos los canales agregados de mentira
        new_data = new_data[:, :, 0]

    # Indicadores para determinar si la imagen 1 esta en la 2
    result = ImageChannelTransformation({
        'Total Keypoints'   : len(kp1),
        'Total Matches'     : len(matches),
        'Match Percentage'  : Measurement(100 * round(len(matches) / len(kp1), 3), '%'),
        'Best Match Distance': round(matches[0].distance, 2)
    }, {})

    return new_data, result

# ******************* Export Functions ********************** #

def add(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.add(first_img.data, second_img.data), []

def sub(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.subtract(first_img.data, second_img.data), []

def multiply(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.multiply(first_img.data, second_img.data), []

def sift(img1: Image, img2: Image, features: int = 0, layers: int = 3, contrast_t: float = 0.04, edge_t: float = 10, sigma: float = 1.6, match_t: float = 200, cross_check: bool = True) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    multi_channel = img1.is_multi_channel()
    color = (255, 255, 255) if multi_channel else (255, 0, 0)
    return img1.combine_over_channels(img2, sift_channel, features, layers, contrast_t, edge_t, sigma, match_t, cross_check, img1.is_multi_channel(), color=color)
