import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx

from src.brain.parietal_lobe.types import Reflex
from src.brain.parietal_lobe.cache import LRUCache
from src.brain.parietal_lobe import reflexes as reflex_module
from src.brain.parietal_lobe import crud_tools
from src.brain.infrastructure.neural_event_bus import NeuralEventBus

logger = logging.getLogger(__name__)

EXPERT_MODELS = [
    "deepseek/deepseek-v3.2",
    "openai/gpt-oss-120b:free",
]


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
        self._register_expert_tool()

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
        """Execute a tool and track usage."""
        if tool_name not in self._reflexes:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        reflex = self._reflexes[tool_name]
        
        # Set activity status
        await NeuralEventBus.set_activity("parietal_lobe", f"Executing: {tool_name}")
        
        try:
            # Track usage
            reflex.usage_count += 1
            reflex.last_used = datetime.now().isoformat()
            
            if asyncio.iscoroutinefunction(reflex.func):
                result = await reflex.func(**args)
            else:
                result = reflex.func(**args)
            
            # Emit tool execution event
            await NeuralEventBus.emit(
                "parietal_lobe", "dashboard", "tool_executed",
                {"tool": tool_name, "usage_count": reflex.usage_count}
            )
            
            await NeuralEventBus.clear_activity("parietal_lobe")
            return result
        except Exception as e:
            await NeuralEventBus.clear_activity("parietal_lobe")
            return json.dumps({"error": f"Error executing {tool_name}: {str(e)}"})

    def get_tools_schema(self) -> List[Dict]:
        return [r.schema for r in self._reflexes.values() if r.enabled]

    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {r.name}: {r.description}" for r in self._reflexes.values() if r.enabled])
    
    def get_tools_stats(self) -> List[Dict]:
        """Get all tools with their usage statistics."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "enabled": r.enabled,
                "usage_count": r.usage_count,
                "last_used": r.last_used
            }
            for r in self._reflexes.values()
        ]

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

    async def _consult_expert(self, task: str, context: str = "") -> str:
        await NeuralEventBus.set_activity("parietal_lobe", "Consulting Expert Model")
        
        if not self._brain or not self._brain.openrouter:
            await NeuralEventBus.clear_activity("parietal_lobe")
            return json.dumps({"error": "Brain/OpenRouter not initialized"})
        
        try:
            # Build expert prompt
            expert_prompt = f"""You are an expert AI consultant helping with complex reasoning tasks.

Task Request:
{task}

{"Context:" + chr(10) + context if context else ""}

Provide a detailed, well-structured response. Be thorough but concise. If this is a planning task, break it into clear steps. If analysis, provide insights and recommendations."""

            # Try expert models in order
            for expert_model in EXPERT_MODELS:
                try:
                    await NeuralEventBus.set_activity("parietal_lobe", f"Expert: {expert_model.split('/')[-1]}")
                    
                    response = await self._brain.openrouter._make_request(
                        model=expert_model,
                        messages=[{"role": "user", "content": expert_prompt}],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    
                    if response and response.get("choices"):
                        result = response["choices"][0]["message"]["content"]
                        await NeuralEventBus.clear_activity("parietal_lobe")
                        
                        return json.dumps({
                            "expert_response": result,
                            "model_used": expert_model,
                            "task": task[:100] + "..." if len(task) > 100 else task
                        })
                        
                except Exception as e:
                    logger.warning(f"Expert model {expert_model} failed: {e}")
                    continue
            
            await NeuralEventBus.clear_activity("parietal_lobe")
            return json.dumps({"error": "All expert models failed. Try again later."})
            
        except Exception as e:
            await NeuralEventBus.clear_activity("parietal_lobe")
            return json.dumps({"error": f"Expert consultation failed: {str(e)}"})

    def _register_expert_tool(self) -> None:
        self.register(Reflex(
            name="consult_expert",
            description="Consult a more powerful AI model for complex reasoning, planning, or analysis tasks. Use when you need deeper thinking or expert-level analysis.",
            func=self._consult_expert,
            schema={
                "type": "function",
                "function": {
                    "name": "consult_expert",
                    "description": "Delegate complex tasks to a more powerful AI model for better reasoning",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task or question to send to the expert model"
                            },
                            "context": {
                                "type": "string", 
                                "description": "Optional additional context to help the expert"
                            }
                        },
                        "required": ["task"]
                    }
                }
            }
        ))
