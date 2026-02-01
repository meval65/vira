from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class SocialIntimacyLayer:
    intimacy_level: float = 0.0
    interaction_history_days: int = 0
    positive_streak: int = 0
    negative_streak: int = 0
    last_interaction: Optional[datetime] = None
    
    INITIAL_FORMALITY_FLOOR: float = 0.6
    MIN_FORMALITY_FLOOR: float = 0.1
    INTIMACY_GROWTH_RATE: float = 0.01
    INTIMACY_DECAY_RATE: float = 0.005
    
    def evolve(self, positive_interaction: bool) -> None:
        if positive_interaction:
            self.positive_streak += 1
            self.negative_streak = 0
            streak_bonus = min(0.02, self.positive_streak * 0.002)
            self.intimacy_level = min(1.0, self.intimacy_level + self.INTIMACY_GROWTH_RATE + streak_bonus)
        else:
            self.negative_streak += 1
            self.positive_streak = 0
            self.intimacy_level = max(0.0, self.intimacy_level - self.INTIMACY_GROWTH_RATE * 2)
        
        self.last_interaction = datetime.now()
    
    def apply_time_decay(self) -> None:
        if not self.last_interaction:
            return
        
        days_since = (datetime.now() - self.last_interaction).days
        if days_since > 0:
            decay = self.INTIMACY_DECAY_RATE * days_since
            self.intimacy_level = max(0.0, self.intimacy_level - decay)
            self.interaction_history_days += days_since
            self.last_interaction = datetime.now()
    
    def get_formality_adjustment(self) -> float:
        range_size = self.INITIAL_FORMALITY_FLOOR - self.MIN_FORMALITY_FLOOR
        return self.INITIAL_FORMALITY_FLOOR - (self.intimacy_level * range_size)
    
    def get_warmth_modifier(self) -> float:
        return self.intimacy_level * 0.3
    
    def as_dict(self) -> Dict:
        return {
            "intimacy_level": self.intimacy_level,
            "interaction_history_days": self.interaction_history_days,
            "positive_streak": self.positive_streak,
            "negative_streak": self.negative_streak,
            "formality_floor": self.get_formality_adjustment()
        }


