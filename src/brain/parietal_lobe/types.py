from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Reflex:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any]
    enabled: bool = True
    usage_count: int = 0
    last_used: Optional[str] = None



