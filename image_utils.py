import numpy as np
from PIL import Image


def image_to_rgba_array(data: np.ndarray) -> np.ndarray:
    shape = data.shape
    if len(shape) == 2:
        # One channel
        return grayscale_to_rgba(data)
    elif len(shape) == 3 and shape[2] == 3:
        # Three channels
        return color_to_rgba(data)

def grayscale_to_rgba(data: np.ndarray) -> np.ndarray:
    # Tenemos que repetir el valor por cada canal de color, y agregar 1 por el canal del alpha
    return color_to_rgba(np.repeat(data.reshape((*data.shape, 1)), 3, axis=2))

def color_to_rgba(data: np.ndarray) -> np.ndarray:
    # Solamente hace falta agregar el canal del alpha (que siempre es 1)
    return np.insert(data, 3, 255, axis=2).flatten() / 255

# height x width x channel
# TODO: Add raw type support
def open_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path))
