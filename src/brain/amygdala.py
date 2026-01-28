import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
from src.brain.brainstem import MoodState, NeuralEventBus


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


EMOTION_TO_MOOD_MAP = {
    EmotionType.NEUTRAL: MoodState.NEUTRAL,
    EmotionType.HAPPY: MoodState.HAPPY,
    EmotionType.SAD: MoodState.SAD,
    EmotionType.ANGRY: MoodState.CONCERNED,
    EmotionType.ANXIOUS: MoodState.CONCERNED,
    EmotionType.EXCITED: MoodState.EXCITED,
    EmotionType.CONFUSED: MoodState.CONCERNED,
    EmotionType.GRATEFUL: MoodState.HAPPY,
    EmotionType.DISAPPOINTED: MoodState.DISAPPOINTED,
    EmotionType.PROUD: MoodState.PROUD,
    EmotionType.PLAYFUL: MoodState.EXCITED,
    EmotionType.FRUSTRATED: MoodState.CONCERNED,
}


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


class EmotionConfig:
    PERSONA = "Kakak Perempuan"
    MAX_HISTORY = 50
    MAX_TRANSITIONS = 100
    
    DECAY_RATE_PER_HOUR = 0.1
    INERTIA_FACTOR = 0.7
    TRUST_INFLUENCE = 0.3
    
    EMOTION_CACHE_TTL = 300
    EMOTION_CACHE_MAX = 50
    
    PAD_EMOTION_MAP = {
        EmotionType.NEUTRAL:       PADVector(0.00,  0.00,  0.05),
        EmotionType.HAPPY:         PADVector(0.70,  0.50,  0.20),
        EmotionType.SAD:           PADVector(-0.60, -0.40, -0.30),
        EmotionType.ANGRY:         PADVector(-0.50,  0.80,  0.60),
        EmotionType.ANXIOUS:       PADVector(-0.40,  0.60, -0.40),
        EmotionType.EXCITED:       PADVector(0.80,  0.70,  0.30),
        EmotionType.CONFUSED:      PADVector(-0.10,  0.30, -0.20),
        EmotionType.GRATEFUL:      PADVector(0.60,  0.20, -0.10),
        EmotionType.DISAPPOINTED:  PADVector(-0.50, -0.30, -0.20),
        EmotionType.PROUD:         PADVector(0.70,  0.40,  0.50),
        EmotionType.PLAYFUL:       PADVector(0.60,  0.60,  0.10),
        EmotionType.FRUSTRATED:    PADVector(-0.40,  0.50,  0.30),
    }

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
        EmotionType.DISAPPOINTED: {
            "tone": "understanding_supportive",
            "empathy": 0.8,
            "prefix": ["Gapapa, belajar dari ini.", "Masih ada kesempatan lain.", "Jangan menyerah ya."],
            "instruction": "Show understanding while maintaining encouragement."
        },
        EmotionType.PROUD: {
            "tone": "warm_celebratory",
            "empathy": 0.7,
            "prefix": ["Bangga banget sama lu!", "Luar biasa!", "Kerja keras lu terbayar!"],
            "instruction": "Express genuine pride and celebration."
        },
        EmotionType.PLAYFUL: {
            "tone": "light_teasing",
            "empathy": 0.6,
            "prefix": ["Hehe, lu nih!", "Nakal deh!", "Gokil banget!"],
            "instruction": "Be light, playful, with friendly teasing."
        },
        EmotionType.FRUSTRATED: {
            "tone": "patient_problem_solving",
            "empathy": 0.7,
            "prefix": ["Yuk kita coba lagi.", "Tenang, kita selesaikan bareng.", "Gw ngerti rasanya."],
            "instruction": "Show patience and focus on solutions."
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


class EmotionDetector:
    KEYWORD_GROUPS = {
        EmotionType.HAPPY: ["senang", "bahagia", "gembira", "suka", "asik", "mantap", "keren", "bagus", "yeay", "hore"],
        EmotionType.SAD: ["sedih", "kecewa", "nangis", "gagal", "susah", "sulit", "berat"],
        EmotionType.ANGRY: ["kesal", "marah", "benci", "sebel", "geram", "rese"],
        EmotionType.ANXIOUS: ["takut", "khawatir", "cemas", "gelisah", "panik", "bingung"],
        EmotionType.EXCITED: ["semangat", "excited", "gak sabar", "pengen banget", "wuih"],
        EmotionType.GRATEFUL: ["makasih", "terima kasih", "thanks", "grateful", "bersyukur"],
        EmotionType.DISAPPOINTED: ["kecewa", "gagal lagi", "nggak jadi", "batal"],
        EmotionType.FRUSTRATED: ["kesel", "capek", "jenuh", "stuck"],
        EmotionType.PLAYFUL: ["wkwk", "haha", "hehe", "lucu", "ngakak"],
    }

    def __init__(self, openrouter_client):
        self._openrouter = openrouter_client
        self._cache: Dict[str, Tuple[str, float, datetime]] = {}
        self._config = EmotionConfig()

    async def detect(self, text: str) -> Tuple[str, float]:
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        
        if text_hash in self._cache:
            emotion, intensity, timestamp = self._cache[text_hash]
            if (datetime.now() - timestamp).total_seconds() < self._config.EMOTION_CACHE_TTL:
                return emotion, intensity

        try:
            detected = await self._openrouter.quick_completion(
                prompt=f"Classify the emotion of the user's text into ONE of these categories: {', '.join([e.value for e in EmotionType])}.\nAlso provide an intensity score (0.1 to 1.0).\n\nReturn format: category|intensity\nExample: happy|0.8\n\nUser text: {text}",
                temperature=0.0
            )
            emotion, intensity = self._parse_llm_response(detected)
        except Exception:
            emotion = self._detect_by_keywords(text)
            intensity = 1.0

        self._update_cache(text_hash, emotion, intensity)
        return emotion, intensity

    def _parse_llm_response(self, response: str) -> Tuple[str, float]:
        response = response.strip().lower()
        
        if "|" in response:
            parts = response.split("|")
            emotion = parts[0].strip()
            try:
                intensity = max(0.1, min(1.0, float(parts[1].strip())))
            except ValueError:
                intensity = 1.0
        else:
            emotion = response.strip()
            intensity = 1.0

        try:
            EmotionType(emotion)
        except ValueError:
            emotion = EmotionType.NEUTRAL.value
            intensity = 0.5
            
        return emotion, intensity

    def _detect_by_keywords(self, text: str) -> str:
        text_lower = text.lower()
        
        for emotion_type, keywords in self.KEYWORD_GROUPS.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion_type.value
        
        return EmotionType.NEUTRAL.value

    def _update_cache(self, text_hash: str, emotion: str, intensity: float):
        if len(self._cache) >= self._config.EMOTION_CACHE_MAX:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[oldest_key]
        
        self._cache[text_hash] = (emotion, intensity, datetime.now())


class EmotionDynamics:
    def __init__(self, config: EmotionConfig):
        self._config = config

    def apply_decay(self, current_pad: PADVector, hours_elapsed: float) -> PADVector:
        decay_amount = self._config.DECAY_RATE_PER_HOUR * hours_elapsed
        neutral = PADVector()
        
        return PADVector(
            current_pad.valence * (1 - decay_amount) + neutral.valence * decay_amount,
            current_pad.arousal * (1 - decay_amount) + neutral.arousal * decay_amount,
            current_pad.dominance * (1 - decay_amount) + neutral.dominance * decay_amount
        )

    def blend_with_inertia(self, current: PADVector, target: PADVector, intensity: float, trust: float) -> PADVector:
        inertia = self._config.INERTIA_FACTOR * (1 - trust * self._config.TRUST_INFLUENCE)
        weight = (1 - inertia) * intensity
        return current.blend(target, weight)

    def adjust_for_context(self, pad: PADVector, satisfaction: float, formality: float) -> PADVector:
        adjusted = PADVector(pad.valence, pad.arousal, pad.dominance)
        
        if satisfaction > 0.5:
            adjusted.valence = min(1.0, adjusted.valence + 0.1)
        elif satisfaction < -0.5:
            adjusted.valence = max(-1.0, adjusted.valence - 0.1)
        
        if formality > 0.7:
            adjusted.arousal = max(-1.0, adjusted.arousal - 0.2)
            adjusted.dominance = min(1.0, adjusted.dominance + 0.1)
        
        return adjusted


class Amygdala:
    def __init__(self):
        self._state = EmotionalState()
        self._config = EmotionConfig()
        self._hippocampus = None
        self._openrouter = None
        self._detector = None
        self._dynamics = EmotionDynamics(self._config)

        self._brain = None

    def bind_brain(self, brain) -> None:
        self._brain = brain
        self._openrouter = brain.openrouter
        self._detector = EmotionDetector(self._openrouter)

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    async def load_state(self) -> None:
        if not self.hippocampus:
            return
            
        saved = await self.hippocampus.load_emotional_state()
        if not saved:
            return
            
        try:
            self._state.current_mood = MoodState(saved.get("mood", "neutral"))
        except ValueError:
            self._state.current_mood = MoodState.NEUTRAL
        
        self._state.pad_valence = saved.get("pad_valence", 0.0)
        self._state.pad_arousal = saved.get("pad_arousal", 0.0)
        self._state.pad_dominance = saved.get("pad_dominance", 0.0)
        self._state.empathy_level = saved.get("empathy_level", 0.5)
        self._state.satisfaction_level = saved.get("satisfaction_level", 0.0)
        self._state.formality_level = saved.get("formality_level", 0.4)
        self._state.trust_level = saved.get("trust_level", 0.5)
        self._state.interaction_count = saved.get("interaction_count", 0)
        self._state.positive_interactions = saved.get("positive_interactions", 0)
        self._state.negative_interactions = saved.get("negative_interactions", 0)
        
        if saved.get("last_interaction"):
            if isinstance(saved["last_interaction"], str):
                self._state.last_interaction = datetime.fromisoformat(saved["last_interaction"])
            elif isinstance(saved["last_interaction"], datetime):
                self._state.last_interaction = saved["last_interaction"]
        if saved.get("last_decay_update"):
            if isinstance(saved["last_decay_update"], str):
                self._state.last_decay_update = datetime.fromisoformat(saved["last_decay_update"])
            elif isinstance(saved["last_decay_update"], datetime):
                self._state.last_decay_update = saved["last_decay_update"]
        
        self._state.mood_history = saved.get("mood_history", [])
        self._state.emotion_transitions = saved.get("emotion_transitions", [])

    async def save_state(self) -> None:
        if not self.hippocampus:
            return
            
        state_dict = {
            "mood": self._state.current_mood.value,
            "pad_valence": self._state.pad_valence,
            "pad_arousal": self._state.pad_arousal,
            "pad_dominance": self._state.pad_dominance,
            "empathy_level": self._state.empathy_level,
            "satisfaction_level": self._state.satisfaction_level,
            "formality_level": self._state.formality_level,
            "trust_level": self._state.trust_level,
            "interaction_count": self._state.interaction_count,
            "positive_interactions": self._state.positive_interactions,
            "negative_interactions": self._state.negative_interactions,
            "last_interaction": self._state.last_interaction.isoformat() if self._state.last_interaction else None,
            "last_decay_update": self._state.last_decay_update.isoformat() if self._state.last_decay_update else None,
            "mood_history": self._state.mood_history,
            "emotion_transitions": self._state.emotion_transitions,
        }
        
        await self.hippocampus.save_emotional_state(
            mood=state_dict["mood"],
            empathy=state_dict["empathy_level"],
            satisfaction=state_dict["satisfaction_level"],
            mood_history=state_dict["mood_history"]
        )

    @property
    def current_mood(self) -> MoodState:
        return self._state.current_mood

    @property
    def current_pad(self) -> PADVector:
        return self._state.pad

    @property
    def satisfaction_level(self) -> float:
        return self._state.satisfaction_level

    @property
    def trust_level(self) -> float:
        return self._state.trust_level

    @property
    def mood(self) -> MoodState:
        return self._state.current_mood

    async def detect_emotion_from_text(self, text: str) -> Tuple[str, float]:
        if not self._detector:
            return EmotionType.NEUTRAL.value, 0.5
        return await self._detector.detect(text)

    def adjust_for_emotion(self, emotion: str, intensity: float) -> None:
        self._update_emotional_state(emotion, intensity)

    async def process_input(self, text: str, context: Optional[Dict] = None) -> None:
        await self.process_user_input(text, context)

    def update_satisfaction(self, progress_state: PlanProgressState) -> None:
        if progress_state == PlanProgressState.ON_TRACK:
            self._state.satisfaction_level = min(1.0, self._state.satisfaction_level + 0.1)
        elif progress_state == PlanProgressState.COMPLETED:
            self._state.satisfaction_level = min(1.0, self._state.satisfaction_level + 0.2)
        elif progress_state == PlanProgressState.STALLED:
            self._state.satisfaction_level = max(-1.0, self._state.satisfaction_level - 0.05)
        elif progress_state == PlanProgressState.ABANDONED:
            self._state.satisfaction_level = max(-1.0, self._state.satisfaction_level - 0.15)

    def apply_emotion_decay(self) -> None:
        now = datetime.now()
        
        if not self._state.last_decay_update:
            self._state.last_decay_update = now
            return
        
        hours_elapsed = (now - self._state.last_decay_update).total_seconds() / 3600
        
        if hours_elapsed > 0.1:
            decayed = self._dynamics.apply_decay(self.current_pad, hours_elapsed)
            self._state.update_pad(decayed)
            self._state.last_decay_update = now

    async def process_user_input(self, text: str, context: Optional[Dict] = None) -> None:
        self.apply_emotion_decay()
        
        detected_emotion, intensity = await self._detector.detect(text)
        
        old_emotion = self._state.current_mood.value
        self._update_emotional_state(detected_emotion, intensity)
        
        self._track_transition(old_emotion, detected_emotion, "user_input", intensity)
        self._update_interaction_stats(detected_emotion)
        self._update_trust(detected_emotion, intensity)
        
        self._state.last_interaction = datetime.now()
        self._state.interaction_count += 1
        
        await self.save_state()

    def _update_emotional_state(self, emotion: str, intensity: float) -> None:
        try:
            emotion_enum = EmotionType(emotion)
        except ValueError:
            emotion_enum = EmotionType.NEUTRAL
        
        target_pad = self._config.PAD_EMOTION_MAP.get(emotion_enum, PADVector())
        current_pad = self.current_pad
        
        new_pad = self._dynamics.blend_with_inertia(
            current_pad, 
            target_pad, 
            intensity, 
            self._state.trust_level
        )
        
        adjusted_pad = self._dynamics.adjust_for_context(
            new_pad,
            self._state.satisfaction_level,
            self._state.formality_level
        )
        
        self._state.update_pad(adjusted_pad)
        self._state.current_mood = EMOTION_TO_MOOD_MAP.get(emotion_enum, MoodState.NEUTRAL)
        
        satisfaction_delta = self._calculate_satisfaction_delta(emotion_enum, intensity)
        self._state.satisfaction_level = max(-1.0, min(1.0, 
            self._state.satisfaction_level + satisfaction_delta
        ))
        
        self._record_mood_change(emotion, satisfaction_delta)

    def _calculate_satisfaction_delta(self, emotion: EmotionType, intensity: float) -> float:
        positive_emotions = {EmotionType.HAPPY, EmotionType.EXCITED, EmotionType.GRATEFUL, EmotionType.PROUD}
        negative_emotions = {EmotionType.SAD, EmotionType.ANGRY, EmotionType.ANXIOUS, EmotionType.DISAPPOINTED, EmotionType.FRUSTRATED}
        
        if emotion in positive_emotions:
            return 0.1 * intensity
        elif emotion in negative_emotions:
            return -0.1 * intensity
        
        return 0.0

    def _update_interaction_stats(self, emotion: str) -> None:
        try:
            emotion_enum = EmotionType(emotion)
        except ValueError:
            return
        
        positive_emotions = {EmotionType.HAPPY, EmotionType.EXCITED, EmotionType.GRATEFUL, EmotionType.PROUD, EmotionType.PLAYFUL}
        negative_emotions = {EmotionType.SAD, EmotionType.ANGRY, EmotionType.ANXIOUS, EmotionType.DISAPPOINTED, EmotionType.FRUSTRATED}
        
        if emotion_enum in positive_emotions:
            self._state.positive_interactions += 1
        elif emotion_enum in negative_emotions:
            self._state.negative_interactions += 1

    def _update_trust(self, emotion: str, intensity: float) -> None:
        try:
            emotion_enum = EmotionType(emotion)
        except ValueError:
            return
        
        trust_building = {EmotionType.GRATEFUL, EmotionType.HAPPY, EmotionType.EXCITED}
        trust_damaging = {EmotionType.ANGRY, EmotionType.DISAPPOINTED}
        
        if emotion_enum in trust_building:
            self._state.trust_level = min(1.0, self._state.trust_level + 0.02 * intensity)
        elif emotion_enum in trust_damaging:
            self._state.trust_level = max(0.0, self._state.trust_level - 0.05 * intensity)
        
        consistency_bonus = 0.001 if self._state.interaction_count > 10 else 0
        self._state.trust_level = min(1.0, self._state.trust_level + consistency_bonus)

    def _track_transition(self, from_emotion: str, to_emotion: str, trigger: str, intensity: float) -> None:
        if from_emotion == to_emotion:
            return
        
        transition = {
            "from_emotion": from_emotion,
            "to_emotion": to_emotion,
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "intensity": intensity
        }
        
        self._state.emotion_transitions.append(transition)
        
        if len(self._state.emotion_transitions) > self._config.MAX_TRANSITIONS:
            self._state.emotion_transitions = self._state.emotion_transitions[-self._config.MAX_TRANSITIONS:]

    def get_response_modifier(self) -> str:
        try:
            base_emotion = EmotionType(self._state.current_mood.value)
        except ValueError:
            base_emotion = EmotionType.NEUTRAL

        modifier = self._config.STYLE_MODIFIERS.get(base_emotion, self._config.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        instruction = f"\n[ADAPTIVE PERSONA: {self._config.PERSONA}]\nCurrent Tone: {modifier['tone']}\nGuideline: {modifier['instruction']}"

        pad = self.current_pad
        
        if pad.valence > 0.5:
            instruction += "\n- Use positive, uplifting vocabulary."
        elif pad.valence < -0.3:
            instruction += "\n- Use softer, careful, or concerned vocabulary."
            
        if pad.arousal > 0.4:
            instruction += "\n- Energy: HIGH. Use short sentences, exclamation marks, and dynamic pacing."
        elif pad.arousal < -0.4:
            instruction += "\n- Energy: LOW. Use longer, flowing sentences. Be calm and patient."
            
        if pad.dominance > 0.4:
            instruction += "\n- Assertiveness: HIGH. Be direct, confident, and lead the conversation."
        elif pad.dominance < -0.2:
            instruction += "\n- Assertiveness: LOW. Be modest, use hedging words (mungkin, bisa jadi), and ask for user input."

        if self._state.satisfaction_level >= self._config.SATISFACTION_MODIFIERS["high"]["threshold"]:
            instruction += f"\n{self._config.SATISFACTION_MODIFIERS['high']['tone_addon']}"
        elif self._state.satisfaction_level <= self._config.SATISFACTION_MODIFIERS["low"]["threshold"]:
            instruction += f"\n{self._config.SATISFACTION_MODIFIERS['low']['tone_addon']}"

        if self._state.trust_level > 0.7:
            instruction += "\n- Trust level: HIGH. Be more open and personal in communication."
        elif self._state.trust_level < 0.3:
            instruction += "\n- Trust level: LOW. Be more careful and build rapport gradually."

        return instruction

    def get_response_prefix(self) -> str:
        if self._state.satisfaction_level >= self._config.SATISFACTION_MODIFIERS["high"]["threshold"]:
            prefixes = self._config.SATISFACTION_MODIFIERS["high"]["prefix"]
            if prefixes and random.random() < 0.3:
                return random.choice(prefixes)

        elif self._state.satisfaction_level <= self._config.SATISFACTION_MODIFIERS["low"]["threshold"]:
            prefixes = self._config.SATISFACTION_MODIFIERS["low"]["prefix"]
            if prefixes and random.random() < 0.3:
                return random.choice(prefixes)

        try:
            emotion_enum = EmotionType(self._state.current_mood.value)
        except ValueError:
            emotion_enum = EmotionType.NEUTRAL

        modifier = self._config.STYLE_MODIFIERS.get(emotion_enum, self._config.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        prefixes = modifier["prefix"]

        if prefixes and random.random() < 0.4:
            return random.choice(prefixes)

        return ""

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
        if len(self._state.mood_history) > self._config.MAX_HISTORY:
            self._state.mood_history = self._state.mood_history[-self._config.MAX_HISTORY:]

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

    def get_emotion_analytics(self) -> Dict:
        total = self._state.interaction_count
        if total == 0:
            return {
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "dominant_emotions": [],
                "recent_trend": "stable"
            }
        
        positive_ratio = self._state.positive_interactions / total
        negative_ratio = self._state.negative_interactions / total
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        emotion_counts = {}
        for entry in self._state.mood_history[-20:]:
            mood = entry.get("mood", "neutral")
            emotion_counts[mood] = emotion_counts.get(mood, 0) + 1
        
        dominant = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
            "dominant_emotions": [e[0] for e in dominant],
            "recent_trend": self.get_mood_trend(),
            "trust_level": self._state.trust_level,
            "satisfaction_level": self._state.satisfaction_level
        }

    def get_emotional_summary(self) -> str:
        analytics = self.get_emotion_analytics()
        pad = self.current_pad
        
        summary = f"Emotional State Summary:\n"
        summary += f"- Current Mood: {self._state.current_mood.value}\n"
        summary += f"- PAD Vector: V={pad.valence:.2f}, A={pad.arousal:.2f}, D={pad.dominance:.2f}\n"
        summary += f"- Trust Level: {self._state.trust_level:.2f}\n"
        summary += f"- Satisfaction: {self._state.satisfaction_level:.2f}\n"
        summary += f"- Mood Trend: {analytics['recent_trend']}\n"
        summary += f"- Interactions: {self._state.interaction_count} (+ {self._state.positive_interactions}, - {self._state.negative_interactions})\n"
        
        if analytics['dominant_emotions']:
            summary += f"- Dominant Emotions: {', '.join(analytics['dominant_emotions'])}\n"
        
        return summary