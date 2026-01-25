"""
User Profile Manager for Vira AI

Manages user identity information including:
- Telegram Username / Display Name
- Custom full name
- Manual context/notes provided by the user
"""

import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from src.database import DBConnection

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    user_id: str
    telegram_name: Optional[str] = None
    full_name: Optional[str] = None
    additional_info: Optional[str] = None
    last_updated: datetime = datetime.now()

class UserProfileManager:
    def __init__(self, db: DBConnection):
        self.db = db
        self._cache: Dict[str, UserProfile] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize user profile table."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                telegram_name TEXT,
                full_name TEXT,
                additional_info TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, ())
        logger.info("[PROFILE-MGR] Tables initialized")
        
    async def get_profile(self, user_id: str) -> UserProfile:
        """Get user profile from cache or DB."""
        if user_id in self._cache:
            return self._cache[user_id]
            
        async with self._lock:
            row = await self.db.fetchone(
                "SELECT telegram_name, full_name, additional_info, last_updated FROM user_profiles WHERE user_id=?", 
                (user_id,)
            )
            
            if row:
                profile = UserProfile(
                    user_id=user_id,
                    telegram_name=row[0],
                    full_name=row[1],
                    additional_info=row[2],
                    last_updated=row[3]
                )
                self._cache[user_id] = profile
                return profile
            
            # Return empty profile if not found logic handled by caller or return default
            return UserProfile(user_id=user_id)

    async def update_telegram_name(self, user_id: str, name: str):
        """Update the telegram display name."""
        if not name:
            return

        async with self._lock:
            # Upsert
            await self.db.execute("""
                INSERT INTO user_profiles (user_id, telegram_name, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    telegram_name=excluded.telegram_name,
                    last_updated=excluded.last_updated
            """, (user_id, name, datetime.now()))
            
            # Invalidate cache
            if user_id in self._cache:
                self._cache[user_id].telegram_name = name
                self._cache[user_id].last_updated = datetime.now()

    async def set_manual_info(self, user_id: str, info: str):
        """Set additional manual info provided by user."""
        async with self._lock:
            await self.db.execute("""
                INSERT INTO user_profiles (user_id, additional_info, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    additional_info=excluded.additional_info,
                    last_updated=excluded.last_updated
            """, (user_id, info, datetime.now()))
            
            if user_id in self._cache:
                self._cache[user_id].additional_info = info
                self._cache[user_id].last_updated = datetime.now()
