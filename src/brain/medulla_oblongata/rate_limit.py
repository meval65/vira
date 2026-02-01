import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Deque

from src.brain.medulla_oblongata.constants import RATE_LIMIT_MAX, RATE_LIMIT_WINDOW

USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LAST_ACTIVITY: Dict[str, datetime] = {}
RATE_LIMIT_TOKENS: Dict[str, Deque[datetime]] = defaultdict(
    lambda: deque(maxlen=RATE_LIMIT_MAX)
)


async def get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]


def check_rate_limit(user_id: str) -> bool:
    now = datetime.now()
    queue = RATE_LIMIT_TOKENS[user_id]
    while queue and queue[0] < now - timedelta(seconds=RATE_LIMIT_WINDOW):
        queue.popleft()
    if len(queue) < RATE_LIMIT_MAX:
        queue.append(now)
        return True
    return False


def update_activity(user_id: str) -> None:
    USER_LAST_ACTIVITY[user_id] = datetime.now()


