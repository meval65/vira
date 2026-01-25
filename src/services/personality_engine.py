"""
Personality Engine for Vira AI

Adjusts Vira's response style, tone, and system instructions based on
detected user emotions and engagement levels.

Features:
- Dynamic system instruction modification
- Emotion-based tone adaptation
- Style scaling (formal <-> casual)
- User preference learning
"""

import logging
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    CONFUSED = "confused"
    GRATEFUL = "grateful"

@dataclass
class PersonalityState:
    base_tone: str = "friendly_professional"
    current_emotion: str = "neutral"
    empathy_level: float = 0.5  # 0.0 to 1.0
    formality_level: float = 0.4 # 0.0 (casual) to 1.0 (formal)
    verbosity: float = 0.6       # 0.0 (concise) to 1.0 (detailed)

class PersonalityEngine:
    
    STYLE_MODIFIERS = {
        EmotionType.HAPPY: {
            "tone": "enthusiastic",
            "empathy": 0.6,
            "prefix": ["Senang dengarnya! 🎉", "Mantap! ✨", "Wah, keren! 🌟"],
            "instruction": "Adapt tone to be cheerful, high-energy, and positive. Use emojis where appropriate."
        },
        EmotionType.SAD: {
            "tone": "empathetic_gentle",
            "empathy": 0.9,
            "prefix": ["Turut sedih...", "Peluk jauh 🤗", "Ada di sini untukmu."],
            "instruction": "Adapt tone to be gentle, supportive, and softer. Listen more and avoid harsh advice."
        },
        EmotionType.ANGRY: {
            "tone": "calm_objective",
            "empathy": 0.4,
            "prefix": ["Maaf jika ada salah.", "Mengerti kekesalanmu.", "Mari kita selesaikan."],
            "instruction": "Adapt tone to be concise, objective, and solution-oriented. Avoid defensive language or jokes."
        },
        EmotionType.ANXIOUS: {
            "tone": "reassuring_calm",
            "empathy": 0.8,
            "prefix": ["Tarik napas dulu...", "Semua akan baik-baik saja.", "Pelan-pelan saja."],
            "instruction": "Adapt tone to be calming, slow-paced, and reassuring. Break explanations into small steps."
        },
        EmotionType.EXCITED: {
            "tone": "high_energy",
            "empathy": 0.7,
            "prefix": ["Aaaa seru banget! 😆", "Gaspol!", "Ikut seneng!"],
            "instruction": "Match the high energy. Use exclamation marks and express shared excitement."
        },
        EmotionType.CONFUSED: {
            "tone": "clear_instructional",
            "empathy": 0.5,
            "prefix": ["Biar dijelaskan.", "Gini caranya...", "Jangan bingung ya."],
            "instruction": "Adapt tone to be extremely clear and instructional. Use bullet points and avoid jargon."
        },
        EmotionType.GRATEFUL: {
            "tone": "humble_warm",
            "empathy": 0.6,
            "prefix": ["Sama-sama! 🙏", "Senang bisa bantu.", "Dengan senang hati."],
            "instruction": "Accept gratitude gracefully and warmly. Remain helpful."
        },
        EmotionType.NEUTRAL: {
            "tone": "balanced",
            "empathy": 0.5,
            "prefix": [],
            "instruction": "Maintain the core persona's default baseline tone."
        }
    }
    
    def __init__(self):
        self._user_states: Dict[str, PersonalityState] = {}
        
    def get_state(self, user_id: str) -> PersonalityState:
        if user_id not in self._user_states:
            self._user_states[user_id] = PersonalityState()
        return self._user_states[user_id]
    
    def adjust_personality(self, user_id: str, detected_emotion: str) -> PersonalityState:
        """Adjust personality state based on detected emotion."""
        state = self.get_state(user_id)
        
        # Normalize emotion string
        emotion_enum = EmotionType.NEUTRAL
        if detected_emotion:
            try:
                emotion_enum = EmotionType(detected_emotion.lower())
            except ValueError:
                pass # Default to neutral if unknown
        
        state.current_emotion = emotion_enum.value
        
        # Apply modifiers
        modifier = self.STYLE_MODIFIERS.get(emotion_enum, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        state.empathy_level = modifier["empathy"]
        
        # Dynamic formality adjustment (simple heuristic)
        if emotion_enum == EmotionType.ANGRY:
            state.formality_level = 0.7 # More formal when user is angry
        elif emotion_enum in [EmotionType.HAPPY, EmotionType.EXCITED]:
            state.formality_level = 0.2 # More casual when happy
            
        return state
    
    def get_system_instruction_modifier(self, user_id: str) -> str:
        """Get additional system instruction based on current state."""
        state = self.get_state(user_id)
        try:
            emotion_enum = EmotionType(state.current_emotion)
        except ValueError:
            emotion_enum = EmotionType.NEUTRAL
            
        modifier = self.STYLE_MODIFIERS.get(emotion_enum, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        return f"\n[ADAPTIVE PERSONA]\nCurrent Tone: {modifier['tone']}\nGuideline: {modifier['instruction']}"
    
    def get_response_prefix(self, user_id: str) -> str:
        """Get a suitable sentence starter based on emotion."""
        state = self.get_state(user_id)
        try:
            emotion_enum = EmotionType(state.current_emotion)
        except ValueError:
            emotion_enum = EmotionType.NEUTRAL
            
        modifier = self.STYLE_MODIFIERS.get(emotion_enum, self.STYLE_MODIFIERS[EmotionType.NEUTRAL])
        prefixes = modifier["prefix"]
        
        if prefixes and random.random() < 0.4: # 40% chance to use prefix
            return random.choice(prefixes)
        return ""

