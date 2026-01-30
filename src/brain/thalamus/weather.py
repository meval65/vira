import datetime
from typing import Optional

import httpx

from src.brain.brainstem import METEOSOURCE_API_KEY


class WeatherMixin:
    DEFAULT_LAT: float = -7.6398581
    DEFAULT_LON: float = 112.2395766
    WEATHER_CACHE_TTL: int = 1800

    async def _get_weather(self) -> Optional[str]:
        if self._is_weather_cached():
            return self._weather_cache.get("formatted")

        if not METEOSOURCE_API_KEY:
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"https://www.meteosource.com/api/v1/free/point"
                params = {
                    "lat": self.DEFAULT_LAT,
                    "lon": self.DEFAULT_LON,
                    "sections": "current",
                    "key": METEOSOURCE_API_KEY
                }
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    current = data.get("current", {})
                    temp = current.get("temperature", "?")
                    summary = current.get("summary", "Unknown")
                    formatted = f"{summary}, {temp}°C"
                    self._weather_cache = {"formatted": formatted, "raw": data}
                    self._weather_timestamp = datetime.datetime.now()
                    return formatted
        except Exception:
            pass
        return None

    def _is_weather_cached(self) -> bool:
        if not self._weather_timestamp:
            return False
        age = (datetime.datetime.now() - self._weather_timestamp).total_seconds()
        return age < self.WEATHER_CACHE_TTL
