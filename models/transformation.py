from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Transformation:
    name: str
    properties: Dict[str, Any]

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.properties = kwargs

    def __str__(self) -> str:
        return '[{0}] {1}'.format(self.name, ", ".join([f'{k}={v}' for k,v in self.properties.items()]))
