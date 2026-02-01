import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.brain.brainstem import MoodState
from .types import (
    EmotionalState, PADVector, OCEANPersonality, EmotionType, 
    PlanProgressState, HardwareStatus, EpisodicEmotionalWeight
)
from .constants import (
    EmotionConfig, EMOTION_TO_MOOD_MAP, find_closest_blended_emotion
)
from .intimacy import SocialIntimacyLayer
from .hardware import HardwareMonitor
from .detector import EmotionDetector
from .dynamics import EmotionDynamics

class Amygdala:
    def __init__(self):
        self._state = EmotionalState()
        self._config = EmotionConfig()
        self._detector = None
        self._dynamics = EmotionDynamics(self._config)
        self._ocean = OCEANPersonality()
        self._intimacy = SocialIntimacyLayer()
        self._episodic_weight = EpisodicEmotionalWeight()
        self._brain = None

    def bind_brain(self, brain):
        self._brain = brain
        self._detector = EmotionDetector(self._brain.openrouter)

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    def apply_persona_calibration(self, calibration: Dict) -> None:
        emotional_inertia = calibration.get("emotional_inertia", 0.7)
        base_arousal = calibration.get("base_arousal", 0.0)
        base_valence = calibration.get("base_valence", 0.0)
        
        self._dynamics.set_persona_calibration(emotional_inertia, base_arousal, base_valence)
        
        self._state.pad_arousal = base_arousal
        self._state.pad_valence = base_valence
        
    async def sync_with_persona(self, persona: Dict) -> None:
        if not persona:
            return
        
        calibration = persona.get("calibration", {})
        self.apply_persona_calibration(calibration)
        
        ocean_data = calibration.get("ocean", {})
        if ocean_data:
            self._ocean = OCEANPersonality.from_dict(ocean_data)
            self._state.empathy_level = min(1.0, 0.5 + self._ocean.get_empathy_modifier())
            self._state.formality_level = self._ocean.get_formality_preference()
        
        identity_anchor = calibration.get("identity_anchor", "")
        if identity_anchor:
            self._config.PERSONA = persona.get("name", self._config.PERSONA)
    
    @property
    def ocean(self) -> OCEANPersonality:
        return self._ocean
    
    def set_ocean(self, ocean: OCEANPersonality) -> None:
        self._ocean = ocean
        pad_modifier = self._ocean.to_pad_modifier()
        current_pad = self._state.pad
        new_pad = current_pad.blend(pad_modifier, weight=0.3)
        self._state.update_pad(new_pad)
        
        self._state.empathy_level = min(1.0, 0.5 + self._ocean.get_empathy_modifier())
        self._state.formality_level = self._ocean.get_formality_preference()
    
    def get_emotional_volatility(self) -> float:
        return self._ocean.get_emotional_volatility()
    
    @property
    def intimacy(self) -> SocialIntimacyLayer:
        return self._intimacy
    
    def record_interaction_quality(self, positive: bool) -> None:
        self._intimacy.evolve(positive)
        
        formality_floor = self._intimacy.get_formality_adjustment()
        self._state.formality_level = max(formality_floor, self._state.formality_level * 0.95)
    
    def apply_intimacy_warmth(self) -> None:
        warmth = self._intimacy.get_warmth_modifier()
        if warmth > 0:
            current_pad = self._state.pad
            new_pad = PADVector(
                min(1.0, current_pad.valence + warmth),
                current_pad.arousal,
                current_pad.dominance
            )
            self._state.update_pad(new_pad)
    
    def apply_memory_mood_shift(self, memories: List[Dict]) -> PADVector:
        if not memories:
            return PADVector(0.0, 0.0, 0.0)
        
        memory_impact = self._episodic_weight.weight_recalled_memories(memories)
        
        weight = 0.2 + (self.get_emotional_volatility() * 0.3)
        current_pad = self._state.pad
        new_pad = current_pad.blend(memory_impact, weight=weight)
        self._state.update_pad(new_pad)
        
        return memory_impact
    
    def get_blended_emotion(self) -> Optional[str]:
        return find_closest_blended_emotion(self._state.pad)
    
    def get_emotion_label(self) -> str:
        blended = self.get_blended_emotion()
        if blended:
            return blended
        
        current_pad = self._state.pad
        closest_emotion = EmotionType.NEUTRAL
        closest_distance = float('inf')
        
        for emotion_type, emotion_pad in self._config.PAD_EMOTION_MAP.items():
            distance = current_pad.distance_to(emotion_pad)
            if distance < closest_distance:
                closest_distance = distance
                closest_emotion = emotion_type
        
        return closest_emotion.value

    def get_hardware_mood_modifier(self) -> PADVector:
        status = HardwareMonitor.get_status()
        if not status.is_available:
            return PADVector(0.0, 0.0, 0.0)
        
        valence_mod = 0.0
        arousal_mod = 0.0
        dominance_mod = 0.0
        
        if status.temperature is not None:
            if status.temperature >= 80:
                arousal_mod += 0.3
                valence_mod -= 0.2
            elif status.temperature >= 70:
                arousal_mod += 0.15
        
        if status.cpu_percent >= 90:
            dominance_mod -= 0.2
            arousal_mod += 0.2
        elif status.cpu_percent >= 70:
            arousal_mod += 0.1
        
        if status.ram_percent >= 90:
            valence_mod -= 0.15
        elif status.ram_percent >= 80:
            valence_mod -= 0.05
        
        return PADVector(valence_mod, arousal_mod, dominance_mod)
    
    def apply_hardware_mood(self) -> None:
        modifier = self.get_hardware_mood_modifier()
        if modifier.valence == 0 and modifier.arousal == 0 and modifier.dominance == 0:
            return
        
        current = self._state.pad
        new_pad = PADVector(
            max(-1.0, min(1.0, current.valence + modifier.valence)),
            max(-1.0, min(1.0, current.arousal + modifier.arousal)),
            max(-1.0, min(1.0, current.dominance + modifier.dominance))
        )
        self._state.update_pad(new_pad)
    
    def get_hardware_status(self) -> HardwareStatus:
        return HardwareMonitor.get_status()
    
    def should_daydream(self) -> bool:
        now = datetime.now()
        hour = now.hour
        
        is_night_time = hour >= 22 or hour < 6
        
        is_idle = False
        if self._state.last_interaction:
            hours_since = (now - self._state.last_interaction).total_seconds() / 3600
            is_idle = hours_since >= 2
        
        is_weekend = now.weekday() >= 5
        
        return is_night_time or is_idle or (is_weekend and hour >= 10 and hour <= 14)
    
    async def trigger_daydream(self) -> Optional[Dict]:
        if not self.hippocampus:
            return None
        
        try:
            return await self.hippocampus.run_daydream_cycle()
        except Exception:
            return None

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
        self.apply_hardware_mood()
        
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

        hw_status = self.get_hardware_status()
        if hw_status.is_available:
            hw_mood = hw_status.get_mood_description()
            if hw_mood != "normal":
                if "overheating" in hw_mood:
                    instruction += "\n- System stress: HIGH TEMPERATURE. You feel irritable and want to be more brief."
                if "overwhelmed" in hw_mood:
                    instruction += "\n- System stress: HIGH CPU. You feel overwhelmed, try to focus responses."
                if "mentally_exhausted" in hw_mood:
                    instruction += "\n- System stress: HIGH RAM. You feel mentally tired, may need short breaks."

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

    def predict_next_emotion(self, current_emotion: str) -> str:
        transitions = [t for t in self._state.emotion_transitions if t['from_emotion'] == current_emotion]
        if not transitions:
            return EmotionType.NEUTRAL.value
        
        emotion_counts = {}
        for t in transitions:
            to_emotion = t['to_emotion']
            emotion_counts[to_emotion] = emotion_counts.get(to_emotion, 0) + 1
        
        if not emotion_counts:
            return EmotionType.NEUTRAL.value
        
        return max(emotion_counts, key=emotion_counts.get)


