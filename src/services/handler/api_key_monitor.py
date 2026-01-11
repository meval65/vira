import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class APIKeyHealthMonitor:
    def __init__(self, num_keys: int):
        self.health = {
            i: {
                'healthy': True,
                'failures': 0,
                'last_fail': None,
                'consecutive_success': 0
            } for i in range(num_keys)
        }
        self.lock = threading.Lock()
        self.FAILURE_THRESHOLD = 3
        self.RECOVERY_TIME = 600
        self.SUCCESS_THRESHOLD = 2
    
    def mark_failure(self, key_index: int):
        with self.lock:
            if key_index not in self.health:
                return
            
            self.health[key_index]['failures'] += 1
            self.health[key_index]['last_fail'] = time.time()
            self.health[key_index]['consecutive_success'] = 0
            
            if self.health[key_index]['failures'] >= self.FAILURE_THRESHOLD:
                self.health[key_index]['healthy'] = False
                logger.warning(f"[API-HEALTH] Key {key_index} marked unhealthy after {self.health[key_index]['failures']} failures")
    
    def mark_success(self, key_index: int):
        with self.lock:
            if key_index not in self.health:
                return
            
            self.health[key_index]['consecutive_success'] += 1
            
            if self.health[key_index]['consecutive_success'] >= self.SUCCESS_THRESHOLD:
                self.health[key_index]['failures'] = 0
                if not self.health[key_index]['healthy']:
                    self.health[key_index]['healthy'] = True
                    logger.info(f"[API-HEALTH] Key {key_index} recovered after consecutive successes")
    
    def get_healthy_key(self, current_index: int, total_keys: int) -> Optional[int]:
        with self.lock:
            now = time.time()
            
            for offset in range(total_keys):
                candidate = (current_index + offset) % total_keys
                key_health = self.health.get(candidate)
                
                if not key_health:
                    continue
                
                if key_health['healthy']:
                    return candidate
                
                if key_health['last_fail']:
                    time_since_fail = now - key_health['last_fail']
                    if time_since_fail > self.RECOVERY_TIME:
                        key_health['healthy'] = True
                        key_health['failures'] = 0
                        key_health['consecutive_success'] = 0
                        logger.info(f"[API-HEALTH] Key {candidate} auto-recovered after {int(time_since_fail)}s")
                        return candidate
            
            return None
    
    def get_status(self) -> dict:
        with self.lock:
            return {
                k: {
                    'healthy': v['healthy'],
                    'failures': v['failures'],
                    'consecutive_success': v['consecutive_success']
                } for k, v in self.health.items()
            }