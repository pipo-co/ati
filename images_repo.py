from typing import Dict

from image_utils import Image

# In Memory Repo
_loaded_images: Dict[str, Image] = {}  # Images by name

def contains_image(image_name: str) -> bool:
    return image_name in _loaded_images

def get_image(image_name: str) -> Image:
    return _loaded_images[image_name]

def persist_image(image: Image) -> None:
    _loaded_images[image.name] = image
