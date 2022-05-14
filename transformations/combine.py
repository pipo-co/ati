from typing import Tuple, List
import numpy as np

from models.image import Image, ImageChannelTransformation

def add(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.add(first_img.data, second_img.data), []

def sub(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.subtract(first_img.data, second_img.data), []

def multiply(first_img: Image, second_img: Image) -> Tuple[np.ndarray, List[ImageChannelTransformation]]:
    return np.multiply(first_img.data, second_img.data), []
