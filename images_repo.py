from typing import Dict, List
from image_utils import Image

# In Memory Repo
_loaded_images: Dict[str, Image] = {}  # Images by name

def contains_image(image_name: str) -> bool:
    return image_name in _loaded_images

def get_image(image_name: str) -> Image:
    return _loaded_images[image_name]

def persist_image(image: Image) -> None:
    _loaded_images[image.name] = image

def get_compatible_images(image_name:str) -> List[Image]:
    original_image = get_image(image_name)
    list = [image for image in _loaded_images.values() if image.name != image_name and image.width == original_image.width and image.height == original_image.height]
    return list

