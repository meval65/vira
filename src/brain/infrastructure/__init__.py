# Infrastructure module for Vira Personal Life OS
# Contains: Redis client, Circuit breaker, NeuralEventBus, MongoDB client, Nocturnal Consolidator

from .redis_client import (
    RedisClient, RedisConfig, ConnectionState,
    get_redis_client, init_redis, close_redis
)

from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    CircuitOpenError, get_circuit_breaker, get_all_circuit_breakers
)

from .neural_event_bus import (
    NeuralEventBus, NeuralEventBusRedis, NeuralEvent, EventType,
    get_event_bus, init_event_bus
)

from .mongo_client import (
    MongoDBClient, get_mongo_client, init_mongodb, close_mongodb
)

__all__ = [
    # Redis
    "RedisClient", "RedisConfig", "ConnectionState", 
    "get_redis_client", "init_redis", "close_redis",
    # Circuit Breaker
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitState",
    "CircuitOpenError", "get_circuit_breaker", "get_all_circuit_breakers",
    # Neural Event Bus
    "NeuralEventBus", "NeuralEventBusRedis", "NeuralEvent", "EventType",
    "get_event_bus", "init_event_bus",
    # MongoDB
    "MongoDBClient", "get_mongo_client", "init_mongodb", "close_mongodb",
]
