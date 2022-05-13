from typing import Tuple
import numpy as np

from models.image import Image, ImageTransformation

def add(first_img: Image, second_img: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return np.add(first_img.data, second_img.data), ImageTransformation('add')

def sub(first_img: Image, second_img: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return np.subtract(first_img.data, second_img.data), ImageTransformation('sub')

def multiply(first_img: Image, second_img: Image) -> Tuple[np.ndarray, ImageTransformation]:
    return np.multiply(first_img.data, second_img.data), ImageTransformation('mult')
