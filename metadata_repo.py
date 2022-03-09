from turtle import width
from typing import Dict
from attr import dataclass
import pandas as pd

@dataclass
class Metadata:
    width: int
    height: int

_loaded_metadata: Dict[str, Metadata] = {} # Metadata

def contains_image_metadata(image_name: str) -> bool:
    return image_name.lower() in _loaded_metadata

def get_metadata(image_name:str) -> Metadata:
    return _loaded_metadata[image_name.lower()]

def persist_metadata(image_name:str, width: int, height: int) -> None:
    _loaded_metadata[image_name.lower()] = Metadata(width, height)


def load_metadata(path: str) -> None:
    df = pd.read_csv(path, sep='\t')
    [persist_metadata(name, width, height) for name, width, height in zip(df['name'], df['width'], df['height'])]
    
