"""
System Config Manager for Vira AI

Manages dynamic configuration overrides, specifically for the global INSTRUCTION key.
Allows switching personas by storing entire system prompts in the database and 
hot-swapping them at runtime.

Features:
- Store multiple 'personas' (system instructions)
- Activate a specific persona to override default config
- Fallback to src.config.INSTRUCTION if no override active
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.database import DBConnection
from src.config import INSTRUCTION as DEFAULT_INSTRUCTION

logger = logging.getLogger(__name__)

@dataclass
class PersonaConfig:
    id: int
    name: str # e.g. "Default Vira", "Professional Assistant", "Pirate Vira"
    instruction_text: str
    is_active: bool
    created_at: datetime

class SystemConfigManager:
    def __init__(self, db: DBConnection):
        self.db = db
        self._active_instruction_cache: Optional[str] = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize config override tables."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS system_personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                instruction_text TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, ())
        
        # Ensure only one is active
        await self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_active_persona ON system_personas(is_active) WHERE is_active = 1",
            ()
        )
        logger.info("[CONFIG-MGR] Tables initialized")
        
    async def get_active_instruction(self) -> str:
        """Get current active instruction (or default if none)."""
        if self._active_instruction_cache:
            return self._active_instruction_cache
            
        async with self._lock:
            row = await self.db.fetchone(
                "SELECT instruction_text FROM system_personas WHERE is_active=1", ()
            )
            
            if row:
                self._active_instruction_cache = row[0]
                return row[0]
            
            # Fallback to default config
            self._active_instruction_cache = DEFAULT_INSTRUCTION
            return DEFAULT_INSTRUCTION

    async def create_persona(self, name: str, instruction_text: str) -> bool:
        """Create a new persona."""
        try:
            await self.db.execute("""
                INSERT INTO system_personas (name, instruction_text, is_active)
                VALUES (?, ?, 0)
            """, (name, instruction_text))
            return True
        except Exception as e:
            logger.error(f"[CONFIG-MGR] Create persona failed: {e}")
            return False

    async def set_active_persona(self, name: str) -> bool:
        """Switch active persona by name. Pass 'default' to reset."""
        async with self._lock:
            try:
                # Deactivate all
                await self.db.execute("UPDATE system_personas SET is_active=0", ())
                
                if name.lower() == 'default':
                    self._active_instruction_cache = DEFAULT_INSTRUCTION
                    logger.info("[CONFIG-MGR] Switched to Default config")
                    return True
                
                # Activate specific
                await self.db.execute("""
                    UPDATE system_personas SET is_active=1 WHERE name=?
                """, (name,))
                
                # Verify activation
                row = await self.db.fetchone(
                    "SELECT instruction_text FROM system_personas WHERE is_active=1", ()
                )
                
                if row:
                    self._active_instruction_cache = row[0]
                    logger.info(f"[CONFIG-MGR] Switched active persona to: {name}")
                    return True
                else:
                    logger.warning(f"[CONFIG-MGR] Persona '{name}' not found, falling back to default")
                    self._active_instruction_cache = DEFAULT_INSTRUCTION
                    return False
                    
            except Exception as e:
                logger.error(f"[CONFIG-MGR] Switch persona failed: {e}")
                return False

    async def list_personas(self) -> List[Dict]:
        """List all available personas."""
        rows = await self.db.fetchall(
            "SELECT id, name, is_active, created_at FROM system_personas ORDER BY name",
            ()
        )
        return [
            {"id": r[0], "name": r[1], "is_active": bool(r[2]), "created_at": r[3]} 
            for r in rows
        ]

    async def update_persona_instruction(self, name: str, new_text: str) -> bool:
        """Update the text of an existing persona."""
        async with self._lock:
            try:
                await self.db.execute("""
                    UPDATE system_personas SET instruction_text=? WHERE name=?
                """, (new_text, name))
                
                # If updating currently active one, invalidate cache
                active_row = await self.db.fetchone("SELECT name FROM system_personas WHERE is_active=1", ())
                if active_row and active_row[0] == name:
                    self._active_instruction_cache = new_text
                
                return True
            except Exception as e:
                logger.error(f"[CONFIG-MGR] Update persona failed: {e}")
                return False
