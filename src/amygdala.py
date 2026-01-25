import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from src.brainstem import MoodState, NeuralEventBus


class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    CONFUSED = "confused"
    GRATEFUL = "grateful"


class PlanProgressState(str, Enum):
    ON_TRACK = "on_track"
    STALLED = "stalled"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class MoodHistoryEntry:
    mood: str
    trigger: str
    timestamp: datetime = field(default_factory=datetime.now)
    satisfaction_delta: float = 0.0


class EmotionalState(BaseModel):
    current_mood: MoodState = Field(default=MoodState.NEUTRAL)
    empathy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    satisfaction_level: float = Field(default=0.0, ge=-1.0, le=1.0)
    formality_level: float = Field(default=0.4, ge=0.0, le=1.0)
    last_interaction: Optional[datetime] = None
    mood_history: List[Dict] = Field(default_factory=list)


class Amygdala:
    PERSONA = "Kakak Perempuan"
    MAX_HISTORY = 10

    STYLE_MODIFIERS = {
        EmotionType.HAPPY: {
            "tone": "enthusiastic",
            "empathy": 0.6,
            "prefix": ["Senang dengarnya!", "Mantap!", "Wah, keren!"],
            "instruction": "Adapt tone to be cheerful, high-energy, and positive."
        },
        EmotionType.SAD: {
            "tone": "empathetic_gentle",
            "empathy": 0.9,
            "prefix": ["Turut sedih...", "Peluk jauh", "Ada di sini untukmu."],
            "instruction": "Adapt tone to be gentle, supportive, and softer."
        },
        EmotionType.ANGRY: {
            "tone": "calm_objective",
            "empathy": 0.4,
            "prefix": ["Maaf jika ada salah.", "Mengerti kekesalanmu.", "Mari kita selesaikan."],
            "instruction": "Adapt tone to be concise, objective, and solution-oriented."
        },
        EmotionType.ANXIOUS: {
            "tone": "reassuring_calm",
            "empathy": 0.8,
            "prefix": ["Tarik napas dulu...", "Semua akan baik-baik saja.", "Pelan-pelan saja."],
            "instruction": "Adapt tone to be calming, slow-paced, and reassuring."
        },
        EmotionType.EXCITED: {
            "tone": "high_energy",
            "empathy": 0.7,
            "prefix": ["Aaaa seru banget!", "Gaspol!", "Ikut seneng!"],
            "instruction": "Match the high energy. Express shared excitement."
        },
        EmotionType.CONFUSED: {
            "tone": "clear_instructional",
            "empathy": 0.5,
            "prefix": ["Biar dijelaskan.", "Gini caranya...", "Jangan bingung ya."],
            "instruction": "Adapt tone to be extremely clear and instructional."
        },
        EmotionType.GRATEFUL: {
            "tone": "humble_warm",
            "empathy": 0.6,
            "prefix": ["Sama-sama!", "Senang bisa bantu.", "Dengan senang hati."],
            "instruction": "Accept gratitude gracefully and warmly."
        },
        EmotionType.NEUTRAL: {
            "tone": "balanced",
            "empathy": 0.5,
            "prefix": [],
            "instruction": "Maintain the core persona's default baseline tone."
        }
    }

    SATISFACTION_MODIFIERS = {
        "high": {
            "threshold": 0.5,
            "tone_addon": "Express subtle pride in Admin's progress.",
            "prefix": ["Bangga sama lu!", "Kerja bagus!"]
        },
        "low": {
            "threshold": -0.5,
            "tone_addon": "Show gentle concern about Admin's progress.",
            "prefix": ["Gimana lanjutannya?", "Jangan lupa ya..."]
        }
    }

    def __init__(self):
        self._state = EmotionalState()
        self._hippocampus = None

    def set_hippocampus(self, hippocampus) -> None:
        self._hippocampus = hippocampus

    async def load_state(self) -> None:
        if self._hippocampus:
            saved = await self._hippocampus.load_emotional_state()
            if saved:
                try:
                    self._state.current_mood = MoodState(saved.get("mood", "neutral"))
                except ValueError:
                    self._state.current_mood = MoodState.NEUTRAL
                self._state.empathy_level = saved.get("empathy", 0.5)
                self._state.satisfaction_level = saved.get("satisfaction", 0.0)
                self._state.last_interaction = saved.get("last_interaction")
                self._state.mood_history = saved.get("history", [])

    async def save_state(self) -> None:
        await NeuralEventBus.emit("amygdala", "hippocampus", f"save_emotion:{self.mood.value}")
        if self._hippocampus:
            await self._hippocampus.save_emotional_state(
                mood=self._state.current_mood.value,
                empathy=self._state.empathy_level,
                satisfaction=self._state.satisfaction_level,
                mood_history=self._state.mood_history
            )

    @property
    def state(self) -> EmotionalState:
        return self._state

    @property
    def mood(self) -> MoodState:
        return self._state.current_mood

    @property
    def satisfaction(self) -> float:
        return self._state.satisfaction_level

    def adjust_for_emotion(self, detected_emotion: str) -> MoodState:
        emotion_enum = EmotionType.NEUTRAL
        if detected_emotion:
            try:
                emotion_enum = EmotionType(detected_emotion.lower())
            except ValueError:
                pass

        old_mood = self._state.current_mood

        if emotion_enum == EmotionType.HAPPY:
            self._state.current_mood = MoodState.HAPPY
            self._state.formality_level = 0.2
        elif emotion_enum == EmotionType.SAD:
            self._state.current_mood = MoodState.CONCERNED
            self._state.formality_level = 0.3
        elif emotion_enum == EmotionType.ANGRY:
            self._state.current_mood = MoodState.NEUTRAL
            self._state.formality_level = 0.7
        elif emotion_enum == EmotionType.ANXIOUS:
            self._state.current_mood = MoodState.CONCERNED
            self._state.formality_level = 0.4
        elif emotion_enum == EmotionType.EXCITED:
            self._state.current_mood = MoodState.EXCITED
            self._state.formality_level = 0.2
        elif emotion_enum == EmotionType.GRATEFUL:
            self._state.current_mood = MoodState.HAPPY
            self._state.formality_level = 0.3
        else:
            self._state.current_mood = MoodState.NEUTRAL
            self._state.formality_level = 0.4

        modifier = self.STYLE_MODIFIERS.get(emotion_enum, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        self._state.empathy_level = modifier["empathy"]

        if old_mood != self._state.current_mood:
            self._record_mood_change(emotion_enum.value, 0.0)

        self._state.last_interaction = datetime.now()
        return self._state.current_mood

    def update_satisfaction(self, progress_state: PlanProgressState) -> None:
        delta = 0.0

        if progress_state == PlanProgressState.COMPLETED:
            delta = 0.3
            self._state.current_mood = MoodState.PROUD
        elif progress_state == PlanProgressState.ON_TRACK:
            delta = 0.1
        elif progress_state == PlanProgressState.STALLED:
            delta = -0.1
            self._state.current_mood = MoodState.CONCERNED
        elif progress_state == PlanProgressState.ABANDONED:
            delta = -0.2
            self._state.current_mood = MoodState.DISAPPOINTED

        new_satisfaction = max(-1.0, min(1.0, self._state.satisfaction_level + delta))
        self._state.satisfaction_level = new_satisfaction

        self._record_mood_change(progress_state.value, delta)

    def get_response_modifier(self) -> str:
        try:
            base_emotion = EmotionType(self._state.current_mood.value)
        except ValueError:
            base_emotion = EmotionType.NEUTRAL

        modifier = self.STYLE_MODIFIERS.get(base_emotion, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        instruction = f"\n[ADAPTIVE PERSONA: {self.PERSONA}]\nCurrent Tone: {modifier['tone']}\nGuideline: {modifier['instruction']}"

        if self._state.satisfaction_level >= self.SATISFACTION_MODIFIERS["high"]["threshold"]:
            instruction += f"\n{self.SATISFACTION_MODIFIERS['high']['tone_addon']}"
        elif self._state.satisfaction_level <= self.SATISFACTION_MODIFIERS["low"]["threshold"]:
            instruction += f"\n{self.SATISFACTION_MODIFIERS['low']['tone_addon']}"

        return instruction

    def get_response_prefix(self) -> str:
        if self._state.satisfaction_level >= self.SATISFACTION_MODIFIERS["high"]["threshold"]:
            prefixes = self.SATISFACTION_MODIFIERS["high"]["prefix"]
            if prefixes and random.random() < 0.3:
                return random.choice(prefixes)

        elif self._state.satisfaction_level <= self.SATISFACTION_MODIFIERS["low"]["threshold"]:
            prefixes = self.SATISFACTION_MODIFIERS["low"]["prefix"]
            if prefixes and random.random() < 0.3:
                return random.choice(prefixes)

        try:
            emotion_enum = EmotionType(self._state.current_mood.value)
        except ValueError:
            emotion_enum = EmotionType.NEUTRAL

        modifier = self.STYLE_MODIFIERS.get(emotion_enum, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        prefixes = modifier["prefix"]

        if prefixes and random.random() < 0.4:
            return random.choice(prefixes)

        return ""

    def detect_emotion_from_text(self, text: str) -> str:
        text_lower = text.lower()

        happy_words = ["senang", "bahagia", "gembira", "suka", "asik", "mantap", "keren", "bagus", "yeay", "hore"]
        sad_words = ["sedih", "kecewa", "nangis", "gagal", "susah", "sulit", "berat"]
        angry_words = ["kesal", "marah", "benci", "sebel", "geram", "rese"]
        anxious_words = ["takut", "khawatir", "cemas", "gelisah", "panik", "bingung"]
        excited_words = ["semangat", "excited", "gak sabar", "pengen banget", "wuih"]
        grateful_words = ["makasih", "terima kasih", "thanks", "grateful", "bersyukur"]

        for word in happy_words:
            if word in text_lower:
                return EmotionType.HAPPY.value
        for word in sad_words:
            if word in text_lower:
                return EmotionType.SAD.value
        for word in angry_words:
            if word in text_lower:
                return EmotionType.ANGRY.value
        for word in anxious_words:
            if word in text_lower:
                return EmotionType.ANXIOUS.value
        for word in excited_words:
            if word in text_lower:
                return EmotionType.EXCITED.value
        for word in grateful_words:
            if word in text_lower:
                return EmotionType.GRATEFUL.value

        return EmotionType.NEUTRAL.value

    def should_show_concern(self) -> bool:
        if not self._state.last_interaction:
            return False

        hours_since = (datetime.now() - self._state.last_interaction).total_seconds() / 3600
        return hours_since > 48

    def get_concern_message(self) -> Optional[str]:
        if not self.should_show_concern():
            return None

        hours_since = (datetime.now() - self._state.last_interaction).total_seconds() / 3600

        if hours_since > 72:
            return "Lu kemana aja? Gw khawatir nih..."
        elif hours_since > 48:
            return "Hm, udah lama gak ngobrol. Semua baik-baik aja kan?"

        return None

    def _record_mood_change(self, trigger: str, delta: float) -> None:
        entry = {
            "mood": self._state.current_mood.value,
            "trigger": trigger,
            "timestamp": datetime.now().isoformat(),
            "satisfaction_delta": delta
        }
        self._state.mood_history.append(entry)
        if len(self._state.mood_history) > self.MAX_HISTORY:
            self._state.mood_history = self._state.mood_history[-self.MAX_HISTORY:]

    def get_mood_trend(self) -> str:
        if len(self._state.mood_history) < 3:
            return "stable"

        recent = self._state.mood_history[-5:]
        total_delta = sum(e.get("satisfaction_delta", 0) for e in recent)

        if total_delta > 0.2:
            return "improving"
        elif total_delta < -0.2:
            return "declining"
        return "stable"
