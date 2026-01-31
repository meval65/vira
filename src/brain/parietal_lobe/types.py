from typing import Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class Reflex:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any]
    enabled: bool = True
