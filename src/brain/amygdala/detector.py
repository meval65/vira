import hashlib
from datetime import datetime
from typing import Dict, Tuple

from .types import EmotionType
from .constants import EmotionConfig

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


