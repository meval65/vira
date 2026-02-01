import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class TripleRelation(str, Enum):
    HAS = "has"
    IS = "is"
    LIKES = "likes"
    DISLIKES = "dislikes"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"
    KNOWS = "knows"
    RELATED_TO = "related_to"
    CREATED = "created"
    OWNS = "owns"
    MEMBER_OF = "member_of"
    PART_OF = "part_of"
    CAUSES = "causes"
    LOCATED_IN = "located_in"


@dataclass
class Memory:
    id: str
    summary: str
    memory_type: str
    priority: float = 0.5
    confidence: float = 0.5
    fingerprint: Optional[str] = None
    entity: Optional[str] = None
    relation: Optional[str] = None
    value: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    use_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
    is_compressed: bool = False
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.priority < 0:
            self.priority = 0.0
        elif self.priority > 1:
            self.priority = 1.0
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0


@dataclass
class Triple:
    id: Optional[str]
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0


class AdminProfile(BaseModel):
    telegram_name: Optional[str] = None
    full_name: Optional[str] = None
    additional_info: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


