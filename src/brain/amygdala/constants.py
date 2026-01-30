from typing import Dict, List, Optional
from src.brain.brainstem import MoodState
from .types import EmotionType, PADVector

BLENDED_EMOTIONS: Dict[str, PADVector] = {
    "remorse": PADVector(-0.3, -0.2, -0.4),
    "optimism": PADVector(0.5, 0.3, 0.2),
    "awe": PADVector(0.4, 0.5, -0.4),
    "contempt": PADVector(-0.2, 0.1, 0.6),
    "love": PADVector(0.8, 0.3, 0.0),
    "submission": PADVector(0.0, -0.3, -0.6),
    "aggressiveness": PADVector(-0.4, 0.8, 0.6),
    "curiosity": PADVector(0.3, 0.5, 0.1),
    
    "nostalgia": PADVector(0.2, -0.3, -0.2),
    "hope": PADVector(0.4, 0.2, 0.1),
    "despair": PADVector(-0.7, -0.5, -0.6),
    "serenity": PADVector(0.4, -0.5, 0.2),
    "vigilance": PADVector(-0.1, 0.7, 0.3),
    "admiration": PADVector(0.6, 0.2, -0.2),
    "terror": PADVector(-0.6, 0.9, -0.8),
    "amazement": PADVector(0.5, 0.7, -0.3),
    "grief": PADVector(-0.8, -0.4, -0.5),
    "loathing": PADVector(-0.5, 0.3, 0.4),
    "ecstasy": PADVector(0.9, 0.8, 0.3),
    "annoyance": PADVector(-0.3, 0.4, 0.2),
    "pensiveness": PADVector(-0.2, -0.4, -0.1),
    "boredom": PADVector(-0.2, -0.6, 0.0),
}

def find_closest_blended_emotion(pad: PADVector, threshold: float = 0.55) -> Optional[str]:
    closest_emotion = None
    closest_distance = float('inf')
    
    for emotion_name, emotion_pad in BLENDED_EMOTIONS.items():
        distance = pad.distance_to(emotion_pad)
        if distance < closest_distance and distance < threshold:
            closest_distance = distance
            closest_emotion = emotion_name
    
    return closest_emotion

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

class EmotionConfig:
    PERSONA = "Kakak Perempuan"
    MAX_HISTORY = 50
    MAX_TRANSITIONS = 100
    
    DECAY_RATE_PER_HOUR = 0.1
    INERTIA_FACTOR = 0.7
    TRUST_INFLUENCE = 0.3
    
    BASE_AROUSAL = 0.0
    BASE_VALENCE = 0.0
    
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
