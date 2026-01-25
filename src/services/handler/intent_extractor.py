import logging
import json
import re
import hashlib
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
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


class IntentPatternLearner:
    def __init__(self):
        self._pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'last_seen': None
        })
        
    def record_pattern(self, intent_type: str, request_type: str, 
                      confidence: float, was_successful: bool):
        pattern_key = f"{intent_type}:{request_type}"
        stats = self._pattern_stats[pattern_key]
        
        stats['total'] += 1
        if was_successful:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
        
        total = stats['total']
        stats['avg_confidence'] = ((stats['avg_confidence'] * (total - 1)) + confidence) / total
        stats['last_seen'] = datetime.now().isoformat()
    
    def get_pattern_reliability(self, intent_type: str, request_type: str) -> float:
        pattern_key = f"{intent_type}:{request_type}"
        stats = self._pattern_stats.get(pattern_key)
        
        if not stats or stats['total'] == 0:
            return 0.5
        
        success_rate = stats['successful'] / stats['total']
        confidence_factor = stats['avg_confidence']
        
        return (success_rate * 0.7) + (confidence_factor * 0.3)
    
    def get_top_patterns(self, limit: int = 10) -> List[Tuple[str, Dict]]:
        sorted_patterns = sorted(
            self._pattern_stats.items(),
            key=lambda x: x[1]['successful'],
            reverse=True
        )
        return sorted_patterns[:limit]


class IntentExtractor:
    MAX_RETRIES = 2
    DEFAULT_TEMPERATURE = 0.1
    MAX_TOKENS = 256
    
    RECURRING_KEYWORDS = {
        'daily': ['setiap hari', 'tiap hari', 'daily', 'harian'],
        'weekly': ['setiap minggu', 'tiap minggu', 'weekly', 'mingguan'],
        'monthly': ['setiap bulan', 'tiap bulan', 'monthly', 'bulanan']
    }
    
    URGENCY_KEYWORDS = {
        'high': ['urgent', 'penting', 'segera', 'cepat', 'asap', 'darurat'],
        'medium': ['soon', 'nanti', 'besok', 'perlu'],
        'low': ['kapan-kapan', 'someday', 'mungkin', 'kalau sempat']
    }
    
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.pattern_learner = IntentPatternLearner()
        self._extraction_history: List[Dict] = []
    
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
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
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
    
    def _detect_recurring_pattern(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        
        for recurring_type, keywords in self.RECURRING_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return recurring_type
        
        return None
    
    def _detect_urgency(self, text: str) -> str:
        text_lower = text.lower()
        
        for urgency_level, keywords in self.URGENCY_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return urgency_level
        
        return 'medium'
    
    def _extract_time_references(self, text: str) -> List[str]:
        time_patterns = [
            r'\d{1,2}:\d{2}',
            r'\d{1,2}\s*(am|pm|pagi|siang|sore|malam)',
            r'(besok|lusa|minggu depan|bulan depan)',
            r'(hari\s+\w+)',
            r'(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)'
        ]
        
        references = []
        text_lower = text.lower()
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                references.extend(matches if isinstance(matches[0], str) else [])
        
        return references[:5]
    
    def _detect_emotion(self, text: str) -> Optional[str]:
        emotion_keywords = {
            'happy': ['senang', 'gembira', 'bahagia', 'excited', 'antusias'],
            'sad': ['sedih', 'kecewa', 'down', 'galau'],
            'angry': ['marah', 'kesal', 'jengkel', 'annoyed'],
            'anxious': ['cemas', 'khawatir', 'nervous', 'takut'],
            'grateful': ['terima kasih', 'thanks', 'grateful', 'appreciate']
        }
        
        text_lower = text.lower()
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return None
    
    def _fallback_extraction(self, user_text: str) -> Dict:
        safe_text = str(user_text) if user_text else ""
        text_lower = safe_text.lower()
        
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
                        'cariin', 'bantu', 'please', 'help']
        if any(word in text_lower for word in request_words):
            intent_type = IntentType.REQUEST.value
            request_type = RequestType.RECOMMENDATION.value
            needs_memory = True
        
        greeting_words = ['halo', 'hai', 'hi', 'hey', 'pagi', 'siang', 'malam', 'hello']
        if any(word in text_lower for word in greeting_words):
            intent_type = IntentType.GREETING.value
            request_type = RequestType.GENERAL_CHAT.value
            needs_memory = False
        
        command_words = ['ingetin', 'jadwalin', 'set', 'atur', 'hapus', 'batalkan', 'remind']
        if any(word in text_lower for word in command_words):
            intent_type = IntentType.COMMAND.value
            request_type = RequestType.SCHEDULE.value
            needs_memory = True
        
        words = re.findall(r'\w+', text_lower)
        entities = [w for w in words if len(w) > 3][:3]
        
        recurring = self._detect_recurring_pattern(safe_text)
        urgency = self._detect_urgency(safe_text)
        time_refs = self._extract_time_references(safe_text)
        emotion = self._detect_emotion(safe_text)
        
        return {
            "intent_type": intent_type,
            "request_type": request_type,
            "entities": entities,
            "key_concepts": [],
            "search_query": safe_text,
            "temporal_context": None,
            "needs_memory": needs_memory,
            "memory_scope": "general" if needs_memory else None,
            "confidence": 0.5,
            "recurring": recurring,
            "urgency": urgency,
            "time_references": time_refs,
            "detected_emotion": emotion,
            "fallback": True
        }
    
    def extract(self, user_text: str) -> Dict[str, Any]:
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
                        
                        extracted['recurring'] = self._detect_recurring_pattern(user_text)
                        extracted['urgency'] = self._detect_urgency(user_text)
                        extracted['time_references'] = self._extract_time_references(user_text)
                        extracted['detected_emotion'] = self._detect_emotion(user_text)
                        
                        reliability = self.pattern_learner.get_pattern_reliability(
                            extracted['intent_type'],
                            extracted['request_type']
                        )
                        extracted['pattern_reliability'] = reliability
                        
                        self._save_to_cache(cache_key, extracted)
                        self._record_extraction(extracted)
                        
                        logger.info(f"[INTENT] Extracted: {extracted['intent_type']} / {extracted['request_type']}")
                        return extracted
            
            except Exception as e:
                logger.warning(f"[INTENT] Extraction attempt {attempt+1} failed: {e}")
        
        logger.warning(f"[INTENT] Using fallback for: {user_text[:50]}")
        fallback = self._fallback_extraction(user_text)
        self._save_to_cache(cache_key, fallback)
        self._record_extraction(fallback)
        return fallback
    
    def _validate_extraction(self, data: Dict) -> bool:
        required = ['intent_type', 'request_type', 'needs_memory']
        return all(field in data for field in required)
    
    def _record_extraction(self, extraction: Dict):
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'intent_type': extraction.get('intent_type'),
            'request_type': extraction.get('request_type'),
            'confidence': extraction.get('confidence', 0.5),
            'fallback': extraction.get('fallback', False)
        }
        
        self._extraction_history.append(history_entry)
        
        if len(self._extraction_history) > 1000:
            self._extraction_history = self._extraction_history[-500:]
    
    def record_extraction_outcome(self, intent_type: str, request_type: str, 
                                 confidence: float, was_successful: bool):
        self.pattern_learner.record_pattern(intent_type, request_type, confidence, was_successful)
    
    def get_extraction_analytics(self) -> Dict:
        if not self._extraction_history:
            return {
                'total_extractions': 0,
                'fallback_rate': 0.0,
                'avg_confidence': 0.0,
                'intent_distribution': {},
                'request_distribution': {}
            }
        
        total = len(self._extraction_history)
        fallback_count = sum(1 for e in self._extraction_history if e.get('fallback'))
        
        intent_counts = defaultdict(int)
        request_counts = defaultdict(int)
        confidences = []
        
        for entry in self._extraction_history:
            intent_counts[entry['intent_type']] += 1
            request_counts[entry['request_type']] += 1
            confidences.append(entry['confidence'])
        
        return {
            'total_extractions': total,
            'fallback_rate': round(fallback_count / total * 100, 2),
            'avg_confidence': round(sum(confidences) / len(confidences), 2),
            'intent_distribution': dict(intent_counts),
            'request_distribution': dict(request_counts),
            'top_patterns': self.pattern_learner.get_top_patterns(5)
        }
    
    def get_cache_stats(self) -> Dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': round(hit_rate, 2)
        }
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0