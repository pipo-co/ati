from typing import Dict, Iterable
from models.image import Image

# In Memory Repo
_loaded_images: Dict[str, Image] = {}  # Images by name

def contains_image(image_name: str) -> bool:
    return image_name in _loaded_images

def get_image(image_name: str) -> Image:
    if image_name not in _loaded_images:
        raise ValueError(f'{image_name} is not in repository')
    return _loaded_images[image_name]

def persist_image(image: Image) -> None:
    if image.name in _loaded_images:
        raise ValueError(f'{image.name} already has mapped {_loaded_images[image.name]}')
    _loaded_images[image.name] = image

def get_images(image_name: str) -> Iterable[Image]:
    original_image = get_image(image_name)
    return (image for image in _loaded_images.values() if image.name != image_name)

def get_same_shape_images(image_name: str) -> Iterable[Image]:
    original_image = get_image(image_name)
    return (image
            for image in _loaded_images.values()
            if image.name != image_name
            and image.shape == original_image.shape
            )
