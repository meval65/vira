"""
Enhanced NeuralEventBus with Redis Pub/Sub for Vira Personal Life OS.

This module provides a Redis-backed event bus that maintains backward compatibility
with the existing NeuralEventBus API while adding:
- Event persistence to MongoDB for audit trail
- Channel-based routing (e.g., vira.hippocampus, vira.amygdala)
- Event replay capability for debugging
- Graceful degradation to in-memory when Redis unavailable
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from src.brain.infrastructure.redis_client import get_redis_client, RedisClient
from src.brain.infrastructure.circuit_breaker import get_circuit_breaker, CircuitBreaker

logger = logging.getLogger(__name__)

# Channel prefix for all Vira events
CHANNEL_PREFIX = "vira"

# Event types
class EventType(str, Enum):
    ACTIVITY_UPDATE = "activity_update"
    SIGNAL = "signal"
    MEMORY = "memory"
    EMOTION = "emotion"
    CONTEXT = "context"
    PERSONA = "persona"
    TOOL = "tool"
    ERROR = "error"


@dataclass
class NeuralEvent:
    """Structured neural event."""
    source: str
    target: str
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuralEvent":
        return cls(
            source=data.get("source", "unknown"),
            target=data.get("target", "broadcast"),
            event_type=data.get("type", "signal"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            event_id=data.get("event_id")
        )


class NeuralEventBusRedis:
    """
    Redis-backed Neural Event Bus with backward-compatible API.
    
    Provides:
    - Pub/Sub messaging across all brain modules
    - Activity tracking for dashboard
    - Event persistence for audit/replay
    - Graceful degradation to in-memory
    """
    
    # Class-level state for backward compatibility
    _current_activity: Dict[str, str] = {}
    _subscribers: List[Callable] = []
    _recent_events: deque = deque(maxlen=100)
    _lock = asyncio.Lock()
    
    # Instance state
    _instance: Optional["NeuralEventBusRedis"] = None
    _redis: Optional[RedisClient] = None
    _circuit_breaker: Optional[CircuitBreaker] = None
    _mongo = None
    _initialized: bool = False
    _channel_subscribers: Dict[str, Set[Callable]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self, mongo_client=None) -> None:
        """Initialize Redis connection and MongoDB for persistence."""
        if self._initialized:
            return
        
        self._redis = get_redis_client()
        self._circuit_breaker = get_circuit_breaker("neural_event_bus")
        self._mongo = mongo_client
        
        # Connect to Redis
        await self._redis.connect()
        
        # Subscribe to broadcast channel
        await self._redis.subscribe(f"{CHANNEL_PREFIX}.*", self._handle_event)
        
        self._initialized = True
        logger.info("NeuralEventBus (Redis) initialized")
    
    async def shutdown(self) -> None:
        """Shutdown event bus."""
        if self._redis:
            await self._redis.close()
        self._initialized = False
        logger.info("NeuralEventBus (Redis) shutdown")
    
    @classmethod
    async def subscribe(cls, callback: Callable) -> None:
        """
        Subscribe to all events (backward compatible).
        
        Args:
            callback: Function to call with event dict
        """
        if callback not in cls._subscribers:
            cls._subscribers.append(callback)
    
    @classmethod
    async def unsubscribe(cls, callback: Callable) -> None:
        """Unsubscribe from events."""
        if callback in cls._subscribers:
            cls._subscribers.remove(callback)
    
    async def subscribe_channel(self, channel: str, callback: Callable) -> None:
        """
        Subscribe to specific channel.
        
        Args:
            channel: Channel name (e.g., "hippocampus", "amygdala")
            callback: Async function to call with event
        """
        full_channel = f"{CHANNEL_PREFIX}.{channel}"
        if full_channel not in self._channel_subscribers:
            self._channel_subscribers[full_channel] = set()
        self._channel_subscribers[full_channel].add(callback)
        
        if self._redis:
            await self._redis.subscribe(full_channel, callback)
    
    @classmethod
    async def set_activity(
        cls,
        module: str,
        description: str,
        payload: Optional[Dict] = None
    ) -> None:
        """
        Set current activity for a module (backward compatible).
        
        Args:
            module: Module name (e.g., "hippocampus")
            description: Activity description
            payload: Optional additional data
        """
        module = module.lower().replace(" ", "_")
        async with cls._lock:
            if cls._current_activity.get(module) != description:
                cls._current_activity[module] = description
        
        await cls.emit(module, "dashboard", EventType.ACTIVITY_UPDATE.value, payload)
    
    @classmethod
    async def clear_activity(cls, module: str) -> None:
        """Clear activity for a module."""
        module = module.lower().replace(" ", "_")
        async with cls._lock:
            cls._current_activity.pop(module, None)
        
        await cls.emit(module, "dashboard", EventType.ACTIVITY_UPDATE.value)
    
    @classmethod
    async def emit(
        cls,
        source: str,
        target: str,
        event_type: str = "signal",
        payload: Optional[Dict] = None
    ) -> None:
        """
        Emit an event (backward compatible).
        
        Args:
            source: Source module name
            target: Target module name or "broadcast"
            event_type: Type of event
            payload: Event data
        """
        import uuid
        
        event = NeuralEvent(
            source=source,
            target=target,
            event_type=event_type,
            payload=payload or {},
            event_id=str(uuid.uuid4())[:8]
        )
        
        event_dict = event.to_dict()
        event_dict["activities"] = cls._current_activity.copy()
        
        # Store in recent events
        cls._recent_events.append(event_dict)
        
        # Publish to Redis if available
        instance = cls._instance
        if instance and instance._redis and instance._initialized:
            try:
                channel = f"{CHANNEL_PREFIX}.{target}"
                await instance._circuit_breaker.call(
                    instance._redis.publish,
                    channel,
                    event_dict,
                    fallback=lambda c, e: None
                )
            except Exception as e:
                logger.debug(f"Redis publish failed, using local dispatch: {e}")
        
        # Always dispatch locally for backward compatibility
        for subscriber in cls._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(cls._safe_call_subscriber(subscriber, event_dict))
                else:
                    subscriber(event_dict)
            except Exception:
                pass
        
        # Persist to MongoDB if available
        if instance and instance._mongo:
            asyncio.create_task(instance._persist_event(event_dict))
    
    @classmethod
    async def _safe_call_subscriber(cls, subscriber: Callable, event: Dict) -> None:
        """Safely call a subscriber."""
        try:
            await subscriber(event)
        except Exception as e:
            logger.error(f"Subscriber error: {e}")
    
    async def _handle_event(self, event_data: Dict) -> None:
        """Handle incoming Redis event."""
        # Add to recent events
        self._recent_events.append(event_data)
        
        # Call channel-specific subscribers
        target = event_data.get("target", "")
        channel = f"{CHANNEL_PREFIX}.{target}"
        
        if channel in self._channel_subscribers:
            for callback in self._channel_subscribers[channel]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(event_data))
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Channel subscriber error: {e}")
        
        # Call global subscribers
        for subscriber in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(self._safe_call_subscriber(subscriber, event_data))
                else:
                    subscriber(event_data)
            except Exception:
                pass
    
    async def _persist_event(self, event: Dict) -> None:
        """Persist event to MongoDB for audit."""
        if not self._mongo:
            return
        
        try:
            await self._mongo.db["event_log"].insert_one({
                **event,
                "persisted_at": datetime.now()
            })
        except Exception as e:
            logger.debug(f"Event persistence failed: {e}")
    
    @classmethod
    def get_recent_events(cls, limit: int = 20) -> List[Dict]:
        """Get recent events."""
        return list(cls._recent_events)[-limit:]
    
    @classmethod
    def get_module_states(cls) -> Dict[str, Any]:
        """Get current module states (backward compatible)."""
        now = datetime.now()
        active_threshold = timedelta(seconds=5)
        
        modules = {
            "brainstem": "idle",
            "hippocampus": "idle",
            "amygdala": "idle",
            "thalamus": "idle",
            "prefrontal_cortex": "idle",
            "motor_cortex": "idle",
            "cerebellum": "idle",
            "occipital_lobe": "active",
            "medulla_oblongata": "idle"
        }
        
        for event in list(cls._recent_events)[-20:]:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                if now - event_time < active_threshold:
                    if event["source"] in modules:
                        modules[event["source"]] = "active"
                    if event["target"] in modules:
                        modules[event["target"]] = "active"
            except (ValueError, KeyError):
                continue
        
        modules["_meta"] = {
            "activities": cls._current_activity.copy(),
            "redis_connected": cls._instance._redis.is_connected if cls._instance and cls._instance._redis else False
        }
        
        return modules
    
    @classmethod
    def get_current_activity(cls) -> Dict[str, str]:
        """Get current activity descriptions for all modules."""
        return cls._current_activity.copy()
    
    async def get_event_history(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get event history from MongoDB.
        
        Args:
            source: Filter by source module
            target: Filter by target module
            event_type: Filter by event type
            limit: Maximum events to return
            since: Only events after this time
            
        Returns:
            List of events
        """
        if not self._mongo:
            return list(self._recent_events)[-limit:]
        
        query = {}
        if source:
            query["source"] = source
        if target:
            query["target"] = target
        if event_type:
            query["type"] = event_type
        if since:
            query["timestamp"] = {"$gte": since.isoformat()}
        
        try:
            cursor = self._mongo.db["event_log"].find(query).sort(
                "timestamp", -1
            ).limit(limit)
            events = await cursor.to_list(length=limit)
            return events
        except Exception as e:
            logger.error(f"Event history query failed: {e}")
            return list(self._recent_events)[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get event bus status."""
        return {
            "initialized": self._initialized,
            "redis_connected": self._redis.is_connected if self._redis else False,
            "redis_fallback": self._redis.is_fallback if self._redis else True,
            "subscribers": len(self._subscribers),
            "channel_subscribers": {k: len(v) for k, v in self._channel_subscribers.items()},
            "recent_event_count": len(self._recent_events),
            "circuit_breaker": self._circuit_breaker.get_status() if self._circuit_breaker else None
        }


# Create backward-compatible class alias
class NeuralEventBus(NeuralEventBusRedis):
    """Backward-compatible alias for NeuralEventBusRedis."""
    pass


# Global instance getter
def get_event_bus() -> NeuralEventBusRedis:
    """Get the NeuralEventBus singleton."""
    return NeuralEventBusRedis()


async def init_event_bus(mongo_client=None) -> NeuralEventBusRedis:
    """Initialize the event bus."""
    bus = get_event_bus()
    await bus.initialize(mongo_client)
    return bus


