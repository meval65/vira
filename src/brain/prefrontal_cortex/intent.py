from datetime import datetime
import hashlib
from typing import Dict, Tuple, Optional, Any

from src.brain.constants import EXTRACTION_INSTRUCTION, UNIFIED_ANALYSIS_INSTRUCTION
from src.brain.prefrontal_cortex.utils import extract_json
from src.brain.prefrontal_cortex.types import IntentType, RequestType

class IntentAnalyzer:
    INTENT_CACHE_TTL = 300

    def __init__(self, openrouter_client, parietal_lobe):
        self._openrouter = openrouter_client
        self.parietal_lobe = parietal_lobe
        self._intent_cache: Dict[str, Tuple[Dict, datetime]] = {}

    async def analyze(self, text: str) -> Dict:
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self._intent_cache:
            cached, timestamp = self._intent_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.INTENT_CACHE_TTL:
                return cached

        try:
            tools_desc = ""
            if self.parietal_lobe:
                tools_desc = self.parietal_lobe.get_tool_descriptions() or "No tools available"
            
            prompt = UNIFIED_ANALYSIS_INSTRUCTION.format(tools_description=tools_desc)
            prompt += f"\n\nUser Input: \"{text}\""
            
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=384,
                temperature=0.1,
                tier="analysis_model",
                json_mode=True
            )
            
            result = extract_json(response)
            if result:
                self._intent_cache[cache_key] = (result, datetime.now())
                return result
            
            return self._fallback_unified_analysis(text)
        except Exception:
            return self._fallback_unified_analysis(text)

    async def extract_intent(self, text: str) -> Dict:
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self._intent_cache:
            cached, timestamp = self._intent_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.INTENT_CACHE_TTL:
                return cached

        try:
            prompt = f"{EXTRACTION_INSTRUCTION}\n\nInput: \"{text}\""
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1,
                tier="analysis_model",
                json_mode=True
            )
            result = extract_json(response) or self._fallback_intent(text)
            self._intent_cache[cache_key] = (result, datetime.now())
            return result
        except Exception:
            return self._fallback_intent(text)

    def _fallback_unified_analysis(self, text: str) -> Dict:
        text_lower = text.lower()
        
        intent_type = "statement"
        if "?" in text or any(w in text_lower for w in ["apa", "siapa", "kapan", "dimana", "gimana", "kenapa", "what", "who", "when", "where", "how", "why"]):
            intent_type = "question"
        elif any(w in text_lower for w in ["ingatkan", "remind", "jadwal", "schedule", "tolong", "please"]):
            intent_type = "request"
        elif any(w in text_lower for w in ["hai", "halo", "hi", "hello", "pagi", "siang", "sore", "malam"]):
            intent_type = "greeting"
        
        request_type = "general_chat"
        if any(w in text_lower for w in ["jadwal", "ingatkan", "remind", "schedule"]):
            request_type = "schedule"
        elif any(w in text_lower for w in ["ingat", "tau", "remember", "know"]):
            request_type = "memory_recall"
        
        sentiment = "neutral"
        if any(w in text_lower for w in ["senang", "happy", "bagus", "mantap", "keren", "suka"]):
            sentiment = "positive"
        elif any(w in text_lower for w in ["sedih", "sad", "kesel", "marah", "benci"]):
            sentiment = "negative"
        
        emotion = "neutral"
        if any(w in text_lower for w in ["senang", "happy", "excited"]):
            emotion = "happy"
        elif any(w in text_lower for w in ["sedih", "sad"]):
            emotion = "sad"
        elif any(w in text_lower for w in ["kesel", "marah", "angry"]):
            emotion = "angry"
        
        entities = []
        words = text.split()
        for w in words:
            if w and w[0].isupper() and len(w) > 2:
                entities.append(w.lower())
        
        return {
            "intent_type": intent_type,
            "request_type": request_type,
            "entities": entities[:5],
            "search_query": text[:100],
            "emotion": emotion,
            "emotion_intensity": 0.5,
            "tool_needed": None,
            "sentiment": sentiment,
            "needs_memory": request_type == "memory_recall",
            "confidence": 0.6
        }

    def _fallback_intent(self, text: str) -> Dict:
        text_lower = text.lower()

        intent_type = IntentType.STATEMENT.value
        if "?" in text or any(w in text_lower for w in ["apa", "siapa", "kapan", "dimana", "gimana", "kenapa", "what", "who", "when", "where", "how", "why"]):
            intent_type = IntentType.QUESTION.value
        elif any(w in text_lower for w in ["ingatkan", "remind", "jadwal", "schedule", "tolong", "please"]):
            intent_type = IntentType.REQUEST.value
        elif any(w in text_lower for w in ["hai", "halo", "hi", "hello", "pagi", "siang", "sore", "malam"]):
            intent_type = IntentType.GREETING.value

        request_type = RequestType.GENERAL_CHAT.value
        if any(w in text_lower for w in ["jadwal", "ingatkan", "remind", "schedule"]):
            request_type = RequestType.SCHEDULE.value
        elif any(w in text_lower for w in ["ingat", "tau", "remember", "know"]):
            request_type = RequestType.MEMORY_RECALL.value

        entities = []
        words = text.split()
        for w in words:
            if w and w[0].isupper() and len(w) > 2:
                entities.append(w.lower())

        return {
            "intent_type": intent_type,
            "request_type": request_type,
            "entities": entities[:5],
            "key_concepts": [],
            "search_query": text[:100],
            "temporal_context": "present",
            "sentiment": "neutral",
            "language": "id",
            "needs_memory": request_type == RequestType.MEMORY_RECALL.value,
            "memory_scope": "personal" if request_type == RequestType.MEMORY_RECALL.value else None,
            "confidence": 0.6
        }
    
    def clear_cache(self) -> int:
        count = len(self._intent_cache)
        self._intent_cache.clear()
        return count


