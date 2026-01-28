import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain.prefrontal_cortex import (
    PrefrontalCortex,
    IntentType,
    RequestType,
    PlanStatus,
    StepStatus,
    TaskPlan,
    TaskStep
)


class TestPrefrontalCortex:

    @pytest.fixture
    def cortex(self):
        pfc = PrefrontalCortex()
        return pfc

    def test_fallback_intent_question(self, cortex):
        result = cortex._fallback_intent("Apa kabar?")
        assert result["intent_type"] == IntentType.QUESTION.value
        assert result["confidence"] == 0.6

    def test_fallback_intent_greeting(self, cortex):
        result = cortex._fallback_intent("Hai Vira!")
        assert result["intent_type"] == IntentType.GREETING.value

    def test_fallback_intent_schedule_request(self, cortex):
        result = cortex._fallback_intent("Ingatkan aku besok jam 9 pagi")
        assert result["intent_type"] == IntentType.REQUEST.value
        assert result["request_type"] == RequestType.SCHEDULE.value

    def test_fallback_intent_memory_recall(self, cortex):
        result = cortex._fallback_intent("Kamu ingat nama kucing saya?")
        assert result["request_type"] == RequestType.MEMORY_RECALL.value
        assert result["needs_memory"] == True

    def test_fallback_intent_statement(self, cortex):
        result = cortex._fallback_intent("Saya suka makan nasi goreng")
        assert result["intent_type"] == IntentType.STATEMENT.value

    def test_fallback_intent_entities_extraction(self, cortex):
        result = cortex._fallback_intent("Kemarin saya bertemu John di Jakarta")
        assert "john" in result["entities"] or "John" in " ".join(result["entities"])

    def test_get_cache_key(self, cortex):
        key1 = cortex._get_cache_key("hello", None)
        key2 = cortex._get_cache_key("hello", None)
        key3 = cortex._get_cache_key("world", None)
        assert key1 == key2
        assert key1 != key3

    def test_get_cache_key_with_image(self, cortex):
        key1 = cortex._get_cache_key("hello", "/path/to/image.jpg")
        key2 = cortex._get_cache_key("hello", None)
        assert key1 != key2


class TestIntentType:

    def test_intent_types_exist(self):
        assert IntentType.QUESTION.value == "question"
        assert IntentType.STATEMENT.value == "statement"
        assert IntentType.REQUEST.value == "request"
        assert IntentType.GREETING.value == "greeting"
        assert IntentType.COMMAND.value == "command"


class TestRequestType:

    def test_request_types_exist(self):
        assert RequestType.INFORMATION.value == "information"
        assert RequestType.SCHEDULE.value == "schedule"
        assert RequestType.MEMORY_RECALL.value == "memory_recall"
        assert RequestType.GENERAL_CHAT.value == "general_chat"


class TestTaskPlan:

    def test_task_plan_creation(self):
        plan = TaskPlan(
            id=1,
            goal="Test goal",
            original_request="Test request"
        )
        assert plan.id == 1
        assert plan.goal == "Test goal"
        assert plan.status == PlanStatus.PENDING
        assert len(plan.steps) == 0

    def test_task_step_creation(self):
        step = TaskStep(
            id=1,
            plan_id=1,
            order_index=0,
            description="Step 1",
            action_type="action"
        )
        assert step.status == StepStatus.PENDING
        assert step.output_result is None


class TestExtractJson:

    @pytest.fixture
    def cortex(self):
        return PrefrontalCortex()

    def test_extract_json_simple(self, cortex):
        text = '{"key": "value"}'
        result = cortex._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_with_markdown(self, cortex):
        text = '```json\n{"key": "value"}\n```'
        result = cortex._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_with_surrounding_text(self, cortex):
        text = 'Here is the result: {"key": "value"} and more text'
        result = cortex._extract_json(text)
        assert result is not None or result == {"key": "value"}

    def test_extract_json_invalid(self, cortex):
        text = "This is not JSON"
        result = cortex._extract_json(text)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
