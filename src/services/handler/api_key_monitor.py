import time
import threading
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class APIKeyHealthMonitor:
    def __init__(self, num_keys: int):
        self.health: Dict[int, Dict] = {
            i: {
                'healthy': True,
                'failures': 0,
                'last_fail': 0.0,
                'consecutive_success': 0
            } for i in range(num_keys)
        }
        self.lock = threading.Lock()
        self.FAILURE_THRESHOLD = 3
        self.RECOVERY_TIME = 300  # Dikurangi ke 5 menit
        self.SUCCESS_THRESHOLD = 2
    
    def mark_failure(self, key_index: int):
        with self.lock:
            if key_index not in self.health:
                return
            
            data = self.health[key_index]
            data['failures'] += 1
            data['last_fail'] = time.time()
            data['consecutive_success'] = 0
            
            if data['failures'] >= self.FAILURE_THRESHOLD:
                data['healthy'] = False
                logger.warning(f"[API-HEALTH] Key {key_index} marked UNHEALTHY (Failures: {data['failures']})")
    
    def mark_success(self, key_index: int):
        with self.lock:
            if key_index not in self.health:
                return
            
            data = self.health[key_index]
            data['consecutive_success'] += 1
            
            if data['consecutive_success'] >= self.SUCCESS_THRESHOLD:
                data['failures'] = 0
                if not data['healthy']:
                    data['healthy'] = True
                    logger.info(f"[API-HEALTH] Key {key_index} RECOVERED")
    
    def get_healthy_key(self, current_index: int, total_keys: int) -> Optional[int]:
        with self.lock:
            now = time.time()
            
            # Cek key lain mulai dari current_index + 1
            for offset in range(total_keys):
                candidate = (current_index + offset) % total_keys
                data = self.health.get(candidate)
                
                if not data:
                    continue
                
                # Jika sehat, langsung pakai
                if data['healthy']:
                    return candidate
                
                # Coba auto-recovery jika sudah melewati masa cool-down
                if now - data['last_fail'] > self.RECOVERY_TIME:
                    data['healthy'] = True
                    data['failures'] = 0
                    data['consecutive_success'] = 0
                    logger.info(f"[API-HEALTH] Key {candidate} AUTO-RECOVERED (Time elapsed)")
                    return candidate
            
            # Emergency: Jika semua mati, cari yang failure count-nya paling sedikit
            logger.error("[API-HEALTH] All keys unhealthy. Forcing best candidate.")
            return min(self.health.keys(), key=lambda k: self.health[k]['failures'])

    def get_status(self) -> dict:
        with self.lock:
            return {k: v.copy() for k, v in self.health.items()}