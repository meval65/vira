from datetime import datetime
from typing import Optional
import psutil
from .types import HardwareStatus

class HardwareMonitor:
    _last_check: Optional[datetime] = None
    _cached_status: Optional[HardwareStatus] = None
    _cache_ttl_seconds: int = 30
    
    @classmethod
    def get_status(cls) -> HardwareStatus:
        now = datetime.now()
        if (cls._cached_status and cls._last_check and 
            (now - cls._last_check).total_seconds() < cls._cache_ttl_seconds):
            return cls._cached_status
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_percent = psutil.virtual_memory().percent
            
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, NotImplementedError):
                pass
            
            cls._cached_status = HardwareStatus(
                cpu_percent=cpu_percent,
                ram_percent=ram_percent,
                temperature=temperature,
                is_available=True
            )
            cls._last_check = now
            return cls._cached_status
        except Exception:
            return HardwareStatus(is_available=False)
