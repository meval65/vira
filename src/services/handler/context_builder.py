import datetime
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextOptimizer:
    def __init__(self):
        self._pattern_cache: Dict[str, int] = defaultdict(int)
        self._effective_patterns: Dict[str, float] = {}
        
    def record_pattern(self, pattern_type: str, was_effective: bool):
        self._pattern_cache[pattern_type] += 1
        
        if pattern_type not in self._effective_patterns:
            self._effective_patterns[pattern_type] = 0.5
        
        current = self._effective_patterns[pattern_type]
        learning_rate = 0.1
        target = 1.0 if was_effective else 0.0
        
        self._effective_patterns[pattern_type] = current + learning_rate * (target - current)
    
    def get_pattern_priority(self, pattern_type: str) -> float:
        return self._effective_patterns.get(pattern_type, 0.5)
    
    def should_include_pattern(self, pattern_type: str, threshold: float = 0.4) -> bool:
        return self.get_pattern_priority(pattern_type) >= threshold


class ContextBuilder:
    def __init__(self, meteosource_api_key: str = None, default_lat: float = -7.6398581, 
                default_lon: float = 112.2395766):
        self.DEFAULT_LAT = default_lat
        self.DEFAULT_LON = default_lon
        self.METEOSOURCE_API_KEY = meteosource_api_key
        
        self._cached_weather: Optional[str] = None
        self._last_weather_fetch: Optional[datetime.datetime] = None
        self._weather_cache_duration = 1800
        
        self.context_cache: Dict[str, Tuple[str, datetime.datetime]] = {}
        self.CACHE_TTL = 300
        self.MAX_CACHE_SIZE = 100
        
        self.optimizer = ContextOptimizer()
        self._context_usage_stats: Dict[str, int] = defaultdict(int)
    
    def _format_time_gap(self, last_time: Optional[datetime.datetime]) -> str:
        if not last_time:
            return "Ini interaksi pertama."
        
        diff = datetime.datetime.now() - last_time
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "Baru saja."
        elif seconds < 3600:
            return f"{int(seconds / 60)} menit lalu."
        elif seconds < 86400:
            return f"{int(seconds / 3600)} jam lalu."
        elif diff.days == 1:
            return "Kemarin."
        elif diff.days < 7:
            return f"{diff.days} hari lalu."
        else:
            return f"{diff.days // 7} minggu lalu."
    
    def _get_weather_data(self) -> str:
        if self._is_weather_cached():
            return self._cached_weather
        
        if not self.METEOSOURCE_API_KEY:
            return "Data cuaca tidak dikonfigurasi."
        
        try:
            return self._fetch_fresh_weather()
        except ImportError:
            return "Module cuaca belum terinstall."
        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")
            return f"{self._cached_weather} (Cached)" if self._cached_weather else "Info cuaca sementara tidak tersedia."
    
    def _is_weather_cached(self) -> bool:
        if not self._cached_weather or not self._last_weather_fetch:
            return False
        elapsed = (datetime.datetime.now() - self._last_weather_fetch).total_seconds()
        return elapsed < self._weather_cache_duration
    
    def _fetch_fresh_weather(self) -> str:
        from pymeteosource.api import Meteosource
        from pymeteosource.types import tiers
        
        ms = Meteosource(self.METEOSOURCE_API_KEY, tiers.FREE)
        forecast = ms.get_point_forecast(
            lat=self.DEFAULT_LAT, 
            lon=self.DEFAULT_LON, 
            sections=['current']
        )
        
        if forecast and hasattr(forecast, 'current'):
            curr = forecast.current
            w_str = f"{curr.summary}, {curr.temperature}°C"
            if hasattr(curr, 'feels_like') and curr.feels_like:
                w_str += f" (terasa {curr.feels_like}°C)"
            
            self._cached_weather = w_str
            self._last_weather_fetch = datetime.datetime.now()
            return w_str
        
        return "Info cuaca sementara tidak tersedia."
    
    def build_context(self, user_id: str, conversation_summary: str,
                     relevant_memories: List[Dict], memory_summary: str,
                     last_interaction: Optional[datetime.datetime], schedule_context: Optional[str],
                     schedule_summary: str, intent_context: Optional[Dict] = None,
                     user_metrics: Optional[Dict] = None, user_profile: Optional[Dict] = None) -> str:
        
        memory_count = len(relevant_memories) if relevant_memories else 0
        state_hash = hashlib.md5(
            f"{conversation_summary}{memory_count}{schedule_context}{schedule_summary}".encode()
        ).hexdigest()[:8]
        cache_key = f"{user_id}:{state_hash}"
        
        # Invalidate cache if profile changes? ideally yes, but for now we assume profile is stable enough or checked outside
        # Actually user_profile should be part of hash if we want robust caching
        
        if not schedule_context and not intent_context and cache_key in self.context_cache:
            cached_ctx, cached_time = self.context_cache[cache_key]
            if (datetime.datetime.now() - cached_time).total_seconds() < self.CACHE_TTL:
                self._context_usage_stats['cache_hits'] += 1
                return cached_ctx
        
        self._context_usage_stats['cache_misses'] += 1
        sections = []
        
        if user_profile:
             profile_section = self._build_profile_section_explicit(user_profile)
             if profile_section:
                 sections.append(profile_section)

        if intent_context:
            intent_section = self._build_intent_section(intent_context)
            if intent_section:
                sections.append(intent_section)
        
        system_section = self._build_system_section(last_interaction, schedule_context)
        sections.append(system_section)
        
        if user_metrics and self.optimizer.should_include_pattern('user_metrics'):
            metrics_section = self._build_metrics_section(user_metrics)
            if metrics_section:
                sections.append(metrics_section)
        
        if conversation_summary:
            sections.append(f"[CONVERSATION CONTEXT]\n{conversation_summary}")
        
        if relevant_memories:
            memory_section = self._build_memory_section(relevant_memories, intent_context)
            sections.append(memory_section)
        
        if memory_summary:
            sections.append(f"[LONG TERM MEMORY SUMMARY]\n{memory_summary}")
        
        if schedule_summary:
            sections.append(f"[UPCOMING SCHEDULES]\n{schedule_summary}")
        
        context = "\n\n".join(sections)
        
        if not schedule_context and not intent_context:
            self.context_cache[cache_key] = (context, datetime.datetime.now())
            self._cleanup_cache()
        
        return context
    
    def _build_profile_section_explicit(self, profile: Dict) -> str:
        """Build explicit user profile section from manual info/telegram name."""
        lines = ["[USER IDENTITY INFO]"]
        
        name = profile.get('telegram_name') or profile.get('full_name')
        if name:
            lines.append(f"Display Name: {name}")
            
        info = profile.get('additional_info')
        if info:
            lines.append(f"User Notes: {info}")
            
        if len(lines) == 1:
            return ""
            
        return "\n".join(lines)

    def _build_metrics_section(self, metrics: Dict) -> str:
        if not metrics or not metrics.get('total_messages'):
            return ""
        
        section = "[USER ENGAGEMENT INSIGHTS]\n"
        
        engagement = metrics.get('engagement_score', 0)
        if engagement > 0.7:
            section += f"Highly engaged user (Score: {engagement:.2f})\n"
        elif engagement > 0.4:
            section += f"Moderately engaged user (Score: {engagement:.2f})\n"
        else:
            section += f"New or casual user (Score: {engagement:.2f})\n"
        
        if metrics.get('total_messages', 0) > 50:
            section += f"Experienced user with {metrics['total_messages']} interactions\n"
        
        top_topics = metrics.get('top_topics', [])
        if top_topics:
            topics_str = ", ".join([topic for topic, _ in top_topics[:3]])
            section += f"Primary interests: {topics_str}\n"
        
        sentiment = metrics.get('avg_sentiment', 0)
        if sentiment > 0.3:
            section += "Generally positive sentiment\n"
        elif sentiment < -0.3:
            section += "May need extra support or encouragement\n"
        
        return section
    
    def _build_intent_section(self, intent_context: Dict) -> str:
        intent_type = intent_context.get('intent_type', 'unknown')
        request_type = intent_context.get('request_type', 'unknown')
        entities = intent_context.get('entities', [])
        key_concepts = intent_context.get('key_concepts', [])
        confidence = intent_context.get('confidence', 0)
        
        section = f"[USER INTENT ANALYSIS]\n"
        section += f"Intent: {intent_type.upper()}\n"
        section += f"Request Type: {request_type.upper()}\n"
        
        if entities:
            section += f"Focus Entities: {', '.join(entities)}\n"
        
        if key_concepts:
            section += f"Key Concepts: {', '.join(key_concepts)}\n"
        
        if confidence:
            section += f"Confidence: {confidence:.2f}\n"
        
        intent_instructions = {
            'question': "User is asking a question. Provide clear, direct answer based on memories.",
            'request': "User is requesting something. Be helpful and actionable.",
            'recommendation': "User wants recommendations. Use preferences and past experiences.",
            'memory_recall': "User is testing/checking memory. Be accurate and reference source.",
            'greeting': "User is greeting. Keep response warm and brief.",
            'command': "User is giving a command. Acknowledge and confirm action.",
            'correction': "User is correcting previous information. Update understanding and confirm.",
            'small_talk': "User is making small talk. Be conversational and natural.",
            'confirmation': "User is confirming something. Acknowledge clearly.",
            'information': "User wants information. Provide clear, factual response.",
            'opinion': "User wants your opinion. Be thoughtful and consider their preferences.",
            'action': "User wants you to do something. Explain what you'll do.",
            'schedule': "User wants to manage schedule. Be clear about what will be scheduled."
        }
        
        instruction = intent_instructions.get(request_type, intent_instructions.get(intent_type, ''))
        if instruction:
            section += f"Response Guidance: {instruction}\n"
        
        return section
    
    def _build_system_section(self, last_interaction: Optional[datetime.datetime],
                             schedule_context: Optional[str]) -> str:
        now_str = datetime.datetime.now().strftime('%A, %d %B %Y, %H:%M')
        weather_info = self._get_weather_data()
        time_gap = self._format_time_gap(last_interaction)
        
        section = f"[SYSTEM CONTEXT]\n"
        section += f"TIME: {now_str}\n"
        section += f"WEATHER: {weather_info}\n"
        section += f"LAST INTERACTION: {time_gap}\n"
        
        if schedule_context:
            section += f"\n⚠️ ACTIVE REMINDER: {schedule_context}"
        
        return section
    
    def _build_memory_section(self, memories: List[Dict], 
                             intent_context: Optional[Dict] = None) -> str:
        if not memories:
            return "[RELEVANT MEMORIES]\nBelum ada memori relevan."
        
        section = "[RELEVANT MEMORIES]\n"
        
        if intent_context:
            request_type = intent_context.get('request_type')
            if request_type == 'memory_recall':
                section += "(User is checking what you remember - be accurate!)\n"
            elif request_type == 'recommendation':
                section += "(Use these preferences to give personalized recommendations)\n"
            elif request_type == 'correction':
                section += "(User may be correcting previous information)\n"
        
        memory_lines = []
        for i, mem in enumerate(memories[:8], 1):
            match_type = mem.get('match_type', 'unknown')
            confidence = mem.get('confidence')
            source_count = mem.get('source_count')
            stability = mem.get('stability_score', 1.0)
            is_volatile = mem.get('volatility_flag', False)
            
            if match_type == "fingerprint_exact":
                prefix = "🎯"
            elif match_type == "entity_relation" or match_type == "entity_match":
                prefix = "🔗"
            else:
                prefix = "💭"
            
            line = f"{prefix} {mem['summary']}"
            
            annotations = []
            
            if match_type == "fingerprint_exact":
                annotations.append("EXACT MATCH")
            
            if confidence and confidence > 0.8:
                annotations.append(f"High Confidence: {confidence:.2f}")
            
            if source_count and source_count > 2:
                annotations.append(f"Mentioned {source_count}x")
            
            if mem.get('priority') and mem['priority'] > 0.8:
                annotations.append("Important")
            
            if stability < 0.5 or is_volatile:
                annotations.append("⚡ Volatile - verify before using")
            elif stability > 0.9:
                annotations.append("✓ Stable")
            
            temporal = mem.get('temporal_context')
            if temporal and temporal != 'unspecified':
                temporal_labels = {
                    'current': 'NOW',
                    'past': 'PAST',
                    'future': 'FUTURE',
                    'permanent': 'ALWAYS'
                }
                annotations.append(temporal_labels.get(temporal, temporal.upper()))
            
            if annotations:
                line += f" [{', '.join(annotations)}]"
            
            memory_lines.append(f"{i}. {line}")
        
        section += "\n".join(memory_lines)
        return section
    
    def _cleanup_cache(self):
        if len(self.context_cache) > self.MAX_CACHE_SIZE:
            sorted_keys = sorted(
                self.context_cache.keys(), 
                key=lambda k: self.context_cache[k][1]
            )
            for k in sorted_keys[:int(self.MAX_CACHE_SIZE * 0.2)]:
                del self.context_cache[k]
    
    def record_context_effectiveness(self, had_intent: bool, had_metrics: bool, was_successful: bool):
        if had_intent:
            self.optimizer.record_pattern('intent_context', was_successful)
        if had_metrics:
            self.optimizer.record_pattern('user_metrics', was_successful)
    
    def clear_cache(self):
        self.context_cache.clear()
        self._cached_weather = None
        self._last_weather_fetch = None
    
    def get_cache_size(self) -> int:
        return len(self.context_cache)
    
    def get_cache_stats(self) -> Dict:
        total_requests = self._context_usage_stats['cache_hits'] + self._context_usage_stats['cache_misses']
        hit_rate = (self._context_usage_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.context_cache),
            'max_size': self.MAX_CACHE_SIZE,
            'ttl_seconds': self.CACHE_TTL,
            'weather_cached': self._cached_weather is not None,
            'cache_hit_rate': round(hit_rate, 2),
            'total_requests': total_requests,
            'optimizer_patterns': dict(self.optimizer._effective_patterns)
        }