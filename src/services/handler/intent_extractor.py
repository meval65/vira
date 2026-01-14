import logging
import json
import re
from typing import Dict, Optional, List, Any
from enum import Enum
from google import genai
from google.genai import types
from src.config import EXTRACTION_INSTRUCTION

logger = logging.getLogger(__name__)


class IntentType(Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    GREETING = "greeting"
    COMMAND = "command"
    SMALL_TALK = "small_talk"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"


class RequestType(Enum):
    INFORMATION = "information"
    RECOMMENDATION = "recommendation"
    MEMORY_RECALL = "memory_recall"
    OPINION = "opinion"
    ACTION = "action"
    SCHEDULE = "schedule"
    GENERAL_CHAT = "general_chat"


class IntentExtractor:
    MAX_RETRIES = 2
    DEFAULT_TEMPERATURE = 0.1
    MAX_TOKENS = 256
    
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        
        try:
            text = text.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON extraction failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in JSON extraction: {e}")
        return None
    
    def _generate_cache_key(self, user_text: str) -> str:
        normalized = user_text.lower().strip()
        return normalized[:100]
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        if len(self._cache) > 500:
            oldest_keys = list(self._cache.keys())[:100]
            for key in oldest_keys:
                del self._cache[key]
        self._cache[cache_key] = data
    
    def _fallback_extraction(self, user_text: str) -> Dict:
        """Simple rule-based fallback if LLM fails"""
        text_lower = user_text.lower()
        
        intent_type = IntentType.STATEMENT.value
        request_type = RequestType.GENERAL_CHAT.value
        needs_memory = False
        
        question_words = ['apa', 'siapa', 'kapan', 'dimana', 'kenapa', 'bagaimana', 
                         'apakah', 'berapa', '?']
        if any(word in text_lower for word in question_words):
            intent_type = IntentType.QUESTION.value
            request_type = RequestType.INFORMATION.value
            needs_memory = True
        
        request_words = ['tolong', 'bisa', 'minta', 'kasih', 'rekomendasiin', 
                        'cariin', 'bantu']
        if any(word in text_lower for word in request_words):
            intent_type = IntentType.REQUEST.value
            request_type = RequestType.RECOMMENDATION.value
            needs_memory = True
        
        greeting_words = ['halo', 'hai', 'hi', 'hey', 'pagi', 'siang', 'malam']
        if any(word in text_lower for word in greeting_words):
            intent_type = IntentType.GREETING.value
            request_type = RequestType.GENERAL_CHAT.value
            needs_memory = False
        
        command_words = ['ingetin', 'jadwalin', 'set', 'atur', 'hapus', 'batalkan']
        if any(word in text_lower for word in command_words):
            intent_type = IntentType.COMMAND.value
            request_type = RequestType.SCHEDULE.value
            needs_memory = True
        
        words = re.findall(r'\w+', text_lower)
        entities = [w for w in words if len(w) > 3][:3]
        
        return {
            "intent_type": intent_type,
            "request_type": request_type,
            "entities": entities,
            "key_concepts": [],
            "search_query": user_text,
            "temporal_context": None,
            "needs_memory": needs_memory,
            "memory_scope": "general" if needs_memory else None,
            "confidence": 0.5,
            "fallback": True
        }
    
    def extract(self, user_text: str) -> Dict[str, Any]:
        """Main extraction method"""
        if not user_text or len(user_text.strip()) < 2:
            return self._fallback_extraction(user_text)
        
        cache_key = self._generate_cache_key(user_text)
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"[INTENT] Cache hit for: {user_text[:50]}")
            return cached
        
        for attempt in range(self.MAX_RETRIES):
            try:
                prompt = f"{EXTRACTION_INSTRUCTION}\n\nInput: {user_text}"
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.DEFAULT_TEMPERATURE,
                        max_output_tokens=self.MAX_TOKENS
                    )
                )
                
                if response.text:
                    extracted = self._extract_json(response.text)
                    
                    if extracted and self._validate_extraction(extracted):
                        extracted['fallback'] = False
                        self._save_to_cache(cache_key, extracted)
                        logger.info(f"[INTENT] Extracted: {extracted['intent_type']} / {extracted['request_type']}")
                        return extracted
            
            except Exception as e:
                logger.warning(f"[INTENT] Extraction attempt {attempt+1} failed: {e}")
        
        logger.warning(f"[INTENT] Using fallback for: {user_text[:50]}")
        fallback = self._fallback_extraction(user_text)
        self._save_to_cache(cache_key, fallback)
        return fallback
    
    def _validate_extraction(self, data: Dict) -> bool:
        """Validate extracted data has required fields"""
        required = ['intent_type', 'request_type', 'needs_memory']
        return all(field in data for field in required)
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': round(hit_rate, 2)
        }
    
    def clear_cache(self):
        """Clear extraction cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0