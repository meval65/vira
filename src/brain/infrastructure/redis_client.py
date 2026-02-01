"""
Redis Client Module for Vira Personal Life OS.

Provides Redis connection management for Pub/Sub event infrastructure with:
- Connection pooling for standard operations and Pub/Sub
- Automatic reconnection with exponential backoff
- Graceful degradation to in-memory fallback
- Health check capabilities

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    REDIS_MAX_CONNECTIONS: Maximum pool connections (default: 10)
"""

import os
import asyncio
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    ConnectionPool = None

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))


class ConnectionState(str, Enum):
    """Redis connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    FALLBACK = "fallback"  # Using in-memory mode


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    url: str = REDIS_URL
    max_connections: int = REDIS_MAX_CONNECTIONS
    db: int = REDIS_DB
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    max_reconnect_attempts: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 30.0


class RedisClient:
    """
    Async Redis client with connection management and fallback capabilities.
    
    Features:
    - Automatic reconnection with exponential backoff
    - In-memory fallback when Redis is unavailable
    - Connection pooling for pub/sub operations
    - Health monitoring
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._client: Optional[aioredis.Redis] = None
        self._pubsub_client: Optional[aioredis.Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._last_health_check: Optional[datetime] = None
        self._subscriptions: Dict[str, Set[Callable]] = {}
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # In-memory fallback storage
        self._fallback_mode = False
        self._fallback_subscribers: Dict[str, List[Callable]] = {}
        self._fallback_events: List[Dict] = []
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._state == ConnectionState.CONNECTED
    
    @property
    def is_fallback(self) -> bool:
        """Check if running in fallback mode."""
        return self._fallback_mode or self._state == ConnectionState.FALLBACK
    
    async def connect(self) -> bool:
        """
        Connect to Redis with fallback to in-memory mode.
        
        Returns:
            True if connected to Redis, False if using fallback mode.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, using in-memory fallback")
            self._enable_fallback()
            return False
        
        async with self._lock:
            if self._state == ConnectionState.CONNECTED:
                return True
            
            self._state = ConnectionState.CONNECTING
            
            try:
                # Create connection pool
                self._pool = ConnectionPool.from_url(
                    self.config.url,
                    max_connections=self.config.max_connections,
                    db=self.config.db,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=True
                )
                
                # Create main client
                self._client = aioredis.Redis(connection_pool=self._pool)
                
                # Test connection
                await self._client.ping()
                
                # Create separate client for pub/sub
                self._pubsub_client = aioredis.Redis(connection_pool=self._pool)
                self._pubsub = self._pubsub_client.pubsub()
                
                self._state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0
                self._fallback_mode = False
                self._last_health_check = datetime.now()
                
                logger.info(f"Redis connected: {self.config.url}")
                return True
                
            except Exception:
                # Log simplified warning without traceback for cleaner startup
                logger.warning("Redis not available, using in-memory fallback")
                self._enable_fallback()
                return False
    
    def _enable_fallback(self) -> None:
        """Enable in-memory fallback mode."""
        self._fallback_mode = True
        self._state = ConnectionState.FALLBACK
        logger.info("Running in in-memory fallback mode")
    
    async def close(self) -> None:
        """Close Redis connections."""
        async with self._lock:
            # Cancel pub/sub listener task
            if self._pubsub_task and not self._pubsub_task.done():
                self._pubsub_task.cancel()
                try:
                    await self._pubsub_task
                except asyncio.CancelledError:
                    pass
            
            # Close pub/sub
            if self._pubsub:
                await self._pubsub.close()
                self._pubsub = None
            
            # Close clients
            if self._pubsub_client:
                await self._pubsub_client.close()
                self._pubsub_client = None
            
            if self._client:
                await self._client.close()
                self._client = None
            
            # Close pool
            if self._pool:
                await self._pool.disconnect()
                self._pool = None
            
            self._state = ConnectionState.DISCONNECTED
            logger.info("Redis connection closed")
    
    async def publish(self, channel: str, event: Dict[str, Any]) -> bool:
        """
        Publish event to a channel.
        
        Args:
            channel: Redis channel name (e.g., "vira.hippocampus")
            event: Event data dictionary
            
        Returns:
            True if published successfully
        """
        event_json = json.dumps(event, default=str)
        
        if self._fallback_mode:
            return await self._fallback_publish(channel, event)
        
        try:
            if not self._client:
                await self.connect()
            
            if self._client:
                await self._client.publish(channel, event_json)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Publish error on {channel}: {e}")
            await self._handle_connection_error()
            return await self._fallback_publish(channel, event)
    
    async def _fallback_publish(self, channel: str, event: Dict[str, Any]) -> bool:
        """Publish event in fallback mode (in-memory)."""
        # Store event
        self._fallback_events.append({
            "channel": channel,
            "event": event,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 events
        if len(self._fallback_events) > 100:
            self._fallback_events = self._fallback_events[-100:]
        
        # Call subscribers directly
        if channel in self._fallback_subscribers:
            for callback in self._fallback_subscribers[channel]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(event))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Fallback subscriber error: {e}")
        
        return True
    
    async def subscribe(self, channel: str, callback: Callable) -> bool:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel pattern (e.g., "vira.*")
            callback: Async function to call with event data
            
        Returns:
            True if subscribed successfully
        """
        if self._fallback_mode:
            if channel not in self._fallback_subscribers:
                self._fallback_subscribers[channel] = []
            self._fallback_subscribers[channel].append(callback)
            return True
        
        try:
            if not self._pubsub:
                await self.connect()
            
            if self._pubsub:
                # Track subscription
                if channel not in self._subscriptions:
                    self._subscriptions[channel] = set()
                self._subscriptions[channel].add(callback)
                
                # Subscribe to channel
                if "*" in channel:
                    await self._pubsub.psubscribe(channel)
                else:
                    await self._pubsub.subscribe(channel)
                
                # Start listener if not running
                if not self._pubsub_task or self._pubsub_task.done():
                    self._pubsub_task = asyncio.create_task(self._listen_pubsub())
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Subscribe error on {channel}: {e}")
            # Fallback to in-memory
            if channel not in self._fallback_subscribers:
                self._fallback_subscribers[channel] = []
            self._fallback_subscribers[channel].append(callback)
            return True
    
    async def _listen_pubsub(self) -> None:
        """Listen for pub/sub messages."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] in ("message", "pmessage"):
                    channel = message.get("channel", message.get("pattern", ""))
                    data = message["data"]
                    
                    try:
                        event = json.loads(data) if isinstance(data, str) else data
                    except json.JSONDecodeError:
                        event = {"raw": data}
                    
                    # Call all matching subscribers
                    for pattern, callbacks in self._subscriptions.items():
                        if self._channel_matches(channel, pattern):
                            for callback in callbacks:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        asyncio.create_task(callback(event))
                                    else:
                                        callback(event)
                                except Exception as e:
                                    logger.error(f"Subscriber callback error: {e}")
                                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Pub/Sub listener error: {e}")
            await self._handle_connection_error()
    
    def _channel_matches(self, channel: str, pattern: str) -> bool:
        """Check if channel matches pattern."""
        if "*" not in pattern:
            return channel == pattern
        
        # Simple glob matching
        import fnmatch
        return fnmatch.fnmatch(channel, pattern)
    
    async def unsubscribe(self, channel: str, callback: Optional[Callable] = None) -> None:
        """Unsubscribe from a channel."""
        if self._fallback_mode:
            if channel in self._fallback_subscribers:
                if callback:
                    self._fallback_subscribers[channel] = [
                        c for c in self._fallback_subscribers[channel] if c != callback
                    ]
                else:
                    del self._fallback_subscribers[channel]
            return
        
        if channel in self._subscriptions:
            if callback:
                self._subscriptions[channel].discard(callback)
                if not self._subscriptions[channel]:
                    del self._subscriptions[channel]
            else:
                del self._subscriptions[channel]
        
        if self._pubsub and channel not in self._subscriptions:
            try:
                if "*" in channel:
                    await self._pubsub.punsubscribe(channel)
                else:
                    await self._pubsub.unsubscribe(channel)
            except Exception:
                pass
    
    async def _handle_connection_error(self) -> None:
        """Handle connection error with reconnection logic."""
        if self._state == ConnectionState.RECONNECTING:
            return
        
        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        
        if self._reconnect_attempts > self.config.max_reconnect_attempts:
            logger.warning("Max reconnection attempts reached, switching to fallback")
            self._enable_fallback()
            return
        
        # Exponential backoff
        delay = min(
            self.config.reconnect_base_delay * (2 ** (self._reconnect_attempts - 1)),
            self.config.reconnect_max_delay
        )
        
        logger.info(f"Reconnecting to Redis in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        
        # Close existing connections
        try:
            if self._client:
                await self._client.close()
            if self._pool:
                await self._pool.disconnect()
        except Exception:
            pass
        
        self._client = None
        self._pool = None
        
        # Attempt reconnection
        await self.connect()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status dictionary
        """
        status = {
            "state": self._state.value,
            "fallback_mode": self._fallback_mode,
            "reconnect_attempts": self._reconnect_attempts,
            "subscriptions": len(self._subscriptions),
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }
        
        if self._client and not self._fallback_mode:
            try:
                await self._client.ping()
                status["ping"] = "ok"
                self._last_health_check = datetime.now()
            except Exception as e:
                status["ping"] = f"error: {e}"
        
        return status
    
    async def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events from fallback storage."""
        return self._fallback_events[-limit:]


# Global instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get the global Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client


async def init_redis() -> RedisClient:
    """Initialize and return the Redis client."""
    client = get_redis_client()
    await client.connect()
    return client


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()


