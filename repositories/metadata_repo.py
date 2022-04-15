import csv
import os
from dataclasses import dataclass
from typing import Dict, Tuple

_METADATA_HEADER: Tuple[str, str, str] = ('name', 'width', 'height')

@dataclass
class Metadata:
    width: int
    height: int

    _METADATA_NAME_IDX:   int = 0
    _METADATA_WIDTH_IDX:  int = 1
    _METADATA_HEIGHT_IDX: int = 2

    @classmethod
    def from_row(cls, row):
        return cls(int(row[cls._METADATA_WIDTH_IDX]), int(row[cls._METADATA_HEIGHT_IDX]))

    @staticmethod
    def name_from_row(row) -> str:
        split_name = os.path.splitext(row[Metadata._METADATA_NAME_IDX])
        return split_name[0] + split_name[1].lower()


_metadata_file_path: str
_loaded_metadata: Dict[str, Metadata] = {}

def set_metadata_file(path: str) -> None:
    global _metadata_file_path, _loaded_metadata
    _metadata_file_path = path
    with open(_metadata_file_path, mode='r') as metadata_file:
        next(metadata_file, None)  # Salteamos el header
        for row in csv.reader(metadata_file, delimiter='\t'):
            _loaded_metadata[Metadata.name_from_row(row)] = Metadata.from_row(row)

def contains_metadata(image_name: str) -> bool:
    return image_name in _loaded_metadata

def get_metadata(image_name: str) -> Metadata:
    return _loaded_metadata[image_name]

def persist_image_metadata(name, width, height) -> None:
    global _metadata_file_path, _loaded_metadata
    _loaded_metadata[name] = Metadata(width, height)
    with open(_metadata_file_path, mode='w') as metadata_file:
        writer = csv.writer(metadata_file, delimiter='\t')
        writer.writerow(_METADATA_HEADER)
        writer.writerows((name, metadata.width, metadata.height) for name, metadata in _loaded_metadata.items())
