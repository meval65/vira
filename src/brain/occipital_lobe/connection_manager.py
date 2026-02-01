import asyncio
import logging
from datetime import datetime
from typing import Set
from fastapi import WebSocket
from collections import deque

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, data: dict):
        message = {"type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.active_connections.discard(connection)

class WebSocketLogHandler(logging.Handler):
    def __init__(self, manager: ConnectionManager, log_buffer: deque):
        super().__init__()
        self.manager = manager
        self.log_buffer = log_buffer

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": self.format(record),
                "source": record.name
            }
            self.log_buffer.append(log_entry)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.broadcast("log", log_entry))
            except RuntimeError:
                pass
        except Exception:
            self.handleError(record)


