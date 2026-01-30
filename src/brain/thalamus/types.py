from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import datetime


class InsightType(str, Enum):
    REMINDER = "reminder"
    FOLLOW_UP = "follow_up"
    PATTERN = "pattern"
    ANNIVERSARY = "anniversary"
    INACTIVITY = "inactivity"
    KNOWLEDGE = "knowledge"
    WELLNESS = "wellness"


class InsightPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class ContextPriority(int, Enum):
    GLOBAL_CONTEXT = 100
    PERSONA_CONTEXT = 90
    SCHEDULE_CONTEXT = 80
    RELEVANT_MEMORIES = 70
    RELEVANT_HISTORY = 60
    INTENT_ANALYSIS = 50
    TIME_CONTEXT = 40
    WEATHER_CONTEXT = 30


@dataclass
class ProactiveInsight:
    insight_type: InsightType
    priority: InsightPriority
    message: str
    context: Dict = field(default_factory=dict)
    trigger_time: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    delivered: bool = False


@dataclass
class SessionMessage:
    role: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    image_path: Optional[str] = None
    embedding: Optional[List[float]] = None
