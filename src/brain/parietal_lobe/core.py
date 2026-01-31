import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

import httpx

from src.brain.parietal_lobe.types import Reflex
from src.brain.parietal_lobe.cache import LRUCache
from src.brain.parietal_lobe import reflexes as reflex_module
from src.brain.parietal_lobe import crud_tools

logger = logging.getLogger(__name__)


class ParietalLobe:
    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 100

    def __init__(self):
        self._brain = None
        self._reflexes: Dict[str, Reflex] = {}
        self._cache = LRUCache(self.MAX_CACHE_SIZE, self.CACHE_TTL_SECONDS)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._safe_math_env = reflex_module.build_safe_math_env()
        self._register_default_reflexes()
        self._register_crud_tools()

    def bind_brain(self, brain) -> None:
        self._brain = brain

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def set_hippocampus(self, hippocampus) -> None:
        self._register_crud_tools()

    def _register_default_reflexes(self) -> None:
        def register(r: Reflex) -> None:
            self._reflexes[r.name] = r

        reflex_module.register_default_reflexes(
            register_fn=register,
            cache=self._cache,
            safe_math_env=self._safe_math_env,
            get_http_client_fn=self._get_http_client,
            get_time_fn=self._get_time,
            calculate_fn=lambda **kw: self._calculate(kw["expression"]),
            get_weather_fn=lambda **kw: self._get_weather(kw["location"]),
            run_python_sandbox_fn=lambda **kw: self._run_python_sandbox(kw["code"]),
        )

    def register(self, reflex: Reflex) -> None:
        self._reflexes[reflex.name] = reflex

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name not in self._reflexes:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        reflex = self._reflexes[tool_name]
        try:
            if asyncio.iscoroutinefunction(reflex.func):
                return await reflex.func(**args)
            return reflex.func(**args)
        except Exception as e:
            return json.dumps({"error": f"Error executing {tool_name}: {str(e)}"})

    def get_tools_schema(self) -> List[Dict]:
        return [r.schema for r in self._reflexes.values() if r.enabled]

    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {r.name}: {r.description}" for r in self._reflexes.values() if r.enabled])

    def _get_time(self) -> str:
        return reflex_module.get_time_impl()

    def _calculate(self, expression: str) -> str:
        return reflex_module.calculate_impl(expression, self._cache, self._safe_math_env)

    async def _get_weather(self, location: str) -> str:
        return await reflex_module.get_weather_impl(location, self._cache, self._get_http_client)

    def _run_python_sandbox(self, code: str) -> str:
        return reflex_module.run_python_sandbox_impl(code, self._cache)

    async def _create_memory(self, summary: str, memory_type: str = "general", priority: float = 0.5) -> str:
        return await crud_tools.create_memory_impl(self.hippocampus, summary=summary, memory_type=memory_type, priority=priority)

    async def _get_memories(self, query: str, limit: int = 5) -> str:
        return await crud_tools.get_memories_impl(self.hippocampus, query=query, limit=limit)

    async def _update_memory(self, memory_id: str, summary: Optional[str] = None, memory_type: Optional[str] = None, priority: Optional[float] = None) -> str:
        return await crud_tools.update_memory_impl(self.hippocampus, memory_id=memory_id, summary=summary, memory_type=memory_type, priority=priority)

    async def _delete_memory(self, memory_id: str) -> str:
        return await crud_tools.delete_memory_impl(self.hippocampus, memory_id=memory_id)

    async def _create_schedule(self, scheduled_at: str, context: str) -> str:
        return await crud_tools.create_schedule_impl(self.hippocampus, scheduled_at=scheduled_at, context=context)

    async def _get_schedules(self, hours_ahead: int = 24, limit: int = 10) -> str:
        return await crud_tools.get_schedules_impl(self.hippocampus, hours_ahead=hours_ahead, limit=limit)

    async def _update_schedule(self, schedule_id: str, scheduled_at: Optional[str] = None, context: Optional[str] = None, status: Optional[str] = None) -> str:
        return await crud_tools.update_schedule_impl(self.hippocampus, schedule_id=schedule_id, scheduled_at=scheduled_at, context=context, status=status)

    async def _delete_schedule(self, schedule_id: str) -> str:
        return await crud_tools.delete_schedule_impl(self.hippocampus, schedule_id=schedule_id)

    def _register_crud_tools(self) -> None:
        crud_reflexes = crud_tools.get_crud_schemas_and_reflexes(
            create_memory_fn=self._create_memory,
            get_memories_fn=self._get_memories,
            update_memory_fn=self._update_memory,
            delete_memory_fn=self._delete_memory,
            create_schedule_fn=self._create_schedule,
            get_schedules_fn=self._get_schedules,
            update_schedule_fn=self._update_schedule,
            delete_schedule_fn=self._delete_schedule,
        )
        for r in crud_reflexes:
            self.register(r)
