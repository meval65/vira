import asyncio
import time
import json
import re
from collections import deque
from functools import wraps
from typing import Callable, Optional, Dict

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
    
    async def acquire(self) -> bool:
        now = time.time()
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def get_wait_time(self) -> float:
        if not self.requests or len(self.requests) < self.max_requests:
            return 0.0
        oldest = self.requests[0]
        return max(0.0, self.window_seconds - (time.time() - oldest))

def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator

def extract_json(text: str) -> Optional[Dict]:
    if not text or not text.strip():
        return None
    
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                end_idx = i + 1
                break
    
    if start_idx >= 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    simple_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if simple_match:
        try:
            return json.loads(simple_match.group())
        except json.JSONDecodeError:
            pass
    
    return None
