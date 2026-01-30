from typing import Optional
from .types import PADVector
from .constants import EmotionConfig

class EmotionDynamics:
    def __init__(self, config: EmotionConfig):
        self._config = config
        self._persona_inertia: Optional[float] = None
        self._persona_base_arousal: float = 0.0
        self._persona_base_valence: float = 0.0

    def set_persona_calibration(self, emotional_inertia: float, base_arousal: float, base_valence: float) -> None:
        self._persona_inertia = max(0.0, min(1.0, emotional_inertia))
        self._persona_base_arousal = max(-1.0, min(1.0, base_arousal))
        self._persona_base_valence = max(-1.0, min(1.0, base_valence))

    def get_effective_inertia(self) -> float:
        if self._persona_inertia is not None:
            return self._persona_inertia
        return self._config.INERTIA_FACTOR

    def apply_decay(self, current_pad: PADVector, hours_elapsed: float) -> PADVector:
        decay_amount = self._config.DECAY_RATE_PER_HOUR * hours_elapsed
        neutral = PADVector(self._persona_base_valence, self._persona_base_arousal, 0.0)
        
        return PADVector(
            current_pad.valence * (1 - decay_amount) + neutral.valence * decay_amount,
            current_pad.arousal * (1 - decay_amount) + neutral.arousal * decay_amount,
            current_pad.dominance * (1 - decay_amount) + neutral.dominance * decay_amount
        )

    def blend_with_inertia(self, current: PADVector, target: PADVector, intensity: float, trust: float) -> PADVector:
        inertia = self.get_effective_inertia()
        trust_adjusted_intensity = intensity * (0.5 + trust * self._config.TRUST_INFLUENCE)
        effective_weight = trust_adjusted_intensity * (1 - inertia)
        return current.blend(target, weight=effective_weight)

    def adjust_for_context(self, pad: PADVector, satisfaction: float, formality: float) -> PADVector:
        satisfaction_mod = satisfaction * 0.1
        formality_mod = (0.5 - formality) * 0.05
        
        return PADVector(
            pad.valence + satisfaction_mod,
            pad.arousal + formality_mod,
            pad.dominance + satisfaction_mod * 0.5
        ).clamp()
