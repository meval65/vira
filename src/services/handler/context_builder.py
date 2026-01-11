import datetime
import hashlib
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ContextBuilder:
    def __init__(self, meteosource_api_key: str = None, default_lat: float = -7.6398581, 
                 default_lon: float = 112.2395766):
        self.DEFAULT_LAT = default_lat
        self.DEFAULT_LON = default_lon
        self.METEOSOURCE_API_KEY = meteosource_api_key
        self._cached_weather = None
        self._last_weather_fetch = None
        self._weather_cache_duration = 900
        self.context_cache: Dict[str, Tuple[str, datetime.datetime]] = {}
        self.CACHE_TTL = 300
        self.MAX_CACHE_SIZE = 100
    
    def _format_time_gap(self, last_time: Optional[datetime.datetime]) -> str:
        if not last_time:
            return "Ini interaksi pertama."
        
        diff = datetime.datetime.now() - last_time
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "Baru saja."
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} menit lalu."
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} jam lalu."
        elif diff.days == 1:
            return "Kemarin."
        elif diff.days < 7:
            return f"{diff.days} hari lalu."
        else:
            weeks = diff.days // 7
            return f"{weeks} minggu lalu."
    
    def _get_weather_data(self) -> str:
        if self._cached_weather and self._last_weather_fetch:
            elapsed = (datetime.datetime.now() - self._last_weather_fetch).total_seconds()
            if elapsed < self._weather_cache_duration:
                return self._cached_weather
        
        if not self.METEOSOURCE_API_KEY:
            return "Data cuaca tidak dikonfigurasi."
        
        try:
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
        
        except Exception as e:
            logger.error(f"[WEATHER-ERROR] {e}")
            if self._cached_weather:
                return f"{self._cached_weather} (Cached)"
        
        return "Cuaca tidak tersedia."
    
    def build_context(self, user_id: str, summary: str, memories: List[str], 
                     memory_summary: str, last_interaction: Optional[datetime.datetime],
                     schedule_context: str = None) -> str:
        
        cache_key = f"{user_id}:{hashlib.md5(summary.encode()).hexdigest()[:8]}"
        
        if not schedule_context and cache_key in self.context_cache:
            cached_ctx, cached_time = self.context_cache[cache_key]
            if (datetime.datetime.now() - cached_time).total_seconds() < self.CACHE_TTL:
                return cached_ctx
        
        now_str = datetime.datetime.now().strftime('%A, %d %B %Y, %H:%M')
        
        mem_str = "\n".join([f"• {m}" for m in memories[:10]]) if memories else "Tidak ada data spesifik."
        
        schedule_alert = f"{schedule_context}\n" if schedule_context else ""
        
        context = (
            f"Gunakan hanya jika relevan sebagai context tambahan\n"
            f"TIME: {now_str}\n"
            f"WEATHER: {self._get_weather_data()}\n"
            f"STATUS: {self._format_time_gap(last_interaction)}\n"
            f"Schedule / Agenda: {schedule_alert}\n"
            f"Conversation Summary: {summary}\n\n"
            f"Long-term Memory Summary: {memory_summary}\n\n"
            f"Relevant Memories: {mem_str}\n\n"
        )
        
        if not schedule_context:
            self.context_cache[cache_key] = (context, datetime.datetime.now())
            self._cleanup_cache()
        
        return context
    
    def _cleanup_cache(self):
        if len(self.context_cache) > self.MAX_CACHE_SIZE:
            oldest_key = min(
                self.context_cache.keys(), 
                key=lambda k: self.context_cache[k][1]
            )
            del self.context_cache[oldest_key]
    
    def clear_cache(self):
        self.context_cache.clear()
        logger.info("[CONTEXT-CACHE] Cache cleared")
    
    def get_cache_size(self) -> int:
        return len(self.context_cache)