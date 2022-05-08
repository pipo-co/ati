import numpy as np

from models.image import Image

def add(first_img: Image, second_img: Image) -> np.ndarray:
    return np.add(first_img.data, second_img.data)

def sub(first_img: Image, second_img: Image) -> np.ndarray:
    return np.subtract(first_img.data, second_img.data)

def multiply(first_img: Image, second_img: Image) -> np.ndarray:
    return np.multiply(first_img.data, second_img.data)
