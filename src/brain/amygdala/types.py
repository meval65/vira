from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.brain.brainstem import MoodState

@dataclass
class PADVector:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def distance_to(self, other: 'PADVector') -> float:
        return ((self.valence - other.valence) ** 2 + 
                (self.arousal - other.arousal) ** 2 + 
                (self.dominance - other.dominance) ** 2) ** 0.5
    
    def blend(self, other: 'PADVector', weight: float = 0.5) -> 'PADVector':
        return PADVector(
            self.valence * (1 - weight) + other.valence * weight,
            self.arousal * (1 - weight) + other.arousal * weight,
            self.dominance * (1 - weight) + other.dominance * weight
        )
    
    def clamp(self, min_val: float = -1.0, max_val: float = 1.0) -> 'PADVector':
        return PADVector(
            max(min_val, min(max_val, self.valence)),
            max(min_val, min(max_val, self.arousal)),
            max(min_val, min(max_val, self.dominance))
        )
    
    def as_list(self) -> List[float]:
        return [self.valence, self.arousal, self.dominance]

    @staticmethod
    def from_list(values: List[float]) -> 'PADVector':
        if len(values) != 3:
            return PADVector()
        return PADVector(values[0], values[1], values[2])

@dataclass
class OCEANPersonality:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    
    def to_pad_modifier(self) -> PADVector:
        valence = (
            (self.extraversion - 0.5) * 0.4 +
            (self.agreeableness - 0.5) * 0.3 -
            (self.neuroticism - 0.5) * 0.5
        )
        arousal = (
            (self.extraversion - 0.5) * 0.3 +
            (self.neuroticism - 0.5) * 0.4 +
            (self.openness - 0.5) * 0.2
        )
        dominance = (
            (self.conscientiousness - 0.5) * 0.3 -
            (self.agreeableness - 0.5) * 0.2
        )
        return PADVector(valence, arousal, dominance)
    
    def get_emotional_volatility(self) -> float:
        return self.neuroticism * 0.6 + (1 - self.conscientiousness) * 0.4
    
    def get_empathy_modifier(self) -> float:
        return self.agreeableness * 0.5 + self.openness * 0.3
    
    def get_formality_preference(self) -> float:
        return self.conscientiousness * 0.5 + (1 - self.extraversion) * 0.3
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism
        }
    
    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'OCEANPersonality':
        return OCEANPersonality(
            openness=data.get("openness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            neuroticism=data.get("neuroticism", 0.5)
        )

class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    CONFUSED = "confused"
    GRATEFUL = "grateful"
    DISAPPOINTED = "disappointed"
    PROUD = "proud"
    PLAYFUL = "playful"
    FRUSTRATED = "frustrated"

class PlanProgressState(str, Enum):
    ON_TRACK = "on_track"
    STALLED = "stalled"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass
class EmotionTransition:
    from_emotion: str
    to_emotion: str
    timestamp: datetime
    trigger: str
    intensity: float

@dataclass
class HardwareStatus:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    temperature: Optional[float] = None
    is_available: bool = False
    
    def get_mood_description(self) -> str:
        parts = []
        if self.temperature and self.temperature >= 80:
            parts.append("overheating")
        elif self.temperature and self.temperature >= 70:
            parts.append("warm")
        if self.cpu_percent >= 90:
            parts.append("overwhelmed")
        elif self.cpu_percent >= 70:
            parts.append("busy")
        if self.ram_percent >= 90:
            parts.append("mentally_exhausted")
        elif self.ram_percent >= 80:
            parts.append("strained")
        return "_".join(parts) if parts else "normal"

@dataclass
class EpisodicEmotionalWeight:
    MEMORY_TYPE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "emotion": 0.8,
        "biography": 0.5,
        "event": 0.6,
        "preference": 0.2,
        "fact": 0.1,
        "decision": 0.4,
        "boundary": 0.3,
        "skill": 0.1,
        "context": 0.2,
    })
    
    RECENCY_DECAY_DAYS: float = 30.0
    
    def calculate_memory_impact(self, memory: Dict) -> PADVector:
        memory_type = memory.get("type", "context")
        base_weight = self.MEMORY_TYPE_WEIGHTS.get(memory_type, 0.2)
        
        created_at = memory.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.now()
            days_old = (datetime.now() - created_at).days
            recency_factor = max(0.1, 1.0 - (days_old / self.RECENCY_DECAY_DAYS))
        else:
            recency_factor = 0.5
        
        sentiment = memory.get("sentiment", "neutral")
        content = memory.get("content", "").lower()
        
        if sentiment == "positive" or any(w in content for w in ["happy", "senang", "bagus", "baik"]):
            base_pad = PADVector(0.3, 0.1, 0.0)
        elif sentiment == "negative" or any(w in content for w in ["sad", "sedih", "marah", "kesal"]):
            base_pad = PADVector(-0.3, 0.1, 0.0)
        else:
            base_pad = PADVector(0.0, 0.0, 0.0)
        
        weight = base_weight * recency_factor
        return PADVector(
            base_pad.valence * weight,
            base_pad.arousal * weight,
            base_pad.dominance * weight
        )
    
    def weight_recalled_memories(self, memories: List[Dict]) -> PADVector:
        if not memories:
            return PADVector(0.0, 0.0, 0.0)
        
        total_valence = 0.0
        total_arousal = 0.0
        total_dominance = 0.0
        
        for memory in memories:
            impact = self.calculate_memory_impact(memory)
            total_valence += impact.valence
            total_arousal += impact.arousal
            total_dominance += impact.dominance
        
        count = len(memories)
        return PADVector(
            max(-1.0, min(1.0, total_valence / count)),
            max(-1.0, min(1.0, total_arousal / count)),
            max(-1.0, min(1.0, total_dominance / count))
        )

class EmotionalState(BaseModel):
    pad_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    pad_arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    pad_dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    current_mood: MoodState = Field(default=MoodState.NEUTRAL)
    
    empathy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    satisfaction_level: float = Field(default=0.0, ge=-1.0, le=1.0)
    formality_level: float = Field(default=0.4, ge=0.0, le=1.0)
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    
    last_interaction: Optional[datetime] = None
    last_decay_update: Optional[datetime] = None
    mood_history: List[Dict] = Field(default_factory=list)
    emotion_transitions: List[Dict] = Field(default_factory=list)

    @property
    def pad(self) -> PADVector:
        return PADVector(self.pad_valence, self.pad_arousal, self.pad_dominance)
    
    def update_pad(self, vector: PADVector):
        clamped = vector.clamp()
        self.pad_valence = clamped.valence
        self.pad_arousal = clamped.arousal
        self.pad_dominance = clamped.dominance


