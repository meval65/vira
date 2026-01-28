import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain.thalamus import Thalamus, SessionMessage, ContextPriority


class TestThalamus:

    @pytest.fixture
    def thalamus(self):
        t = Thalamus()
        return t

    def test_estimate_tokens_empty(self, thalamus):
        assert thalamus.estimate_tokens("") == 0
        assert thalamus.estimate_tokens(None) == 0

    def test_estimate_tokens_short_text(self, thalamus):
        text = "Hello world"
        result = thalamus.estimate_tokens(text)
        assert result >= 2

    def test_estimate_tokens_indonesian_text(self, thalamus):
        text = "Selamat pagi, apa kabar hari ini? Semoga harimu menyenangkan."
        result = thalamus.estimate_tokens(text)
        assert result > 0
        assert result >= len(text.split())

    def test_estimate_tokens_mixed_language(self, thalamus):
        text = "Hey bro, gimana project Python-nya? Udah kelar belum?"
        result = thalamus.estimate_tokens(text)
        words = len(text.split())
        chars = len(text)
        expected = max(words, chars // 3)
        assert result == expected

    def test_cosine_similarity_identical(self, thalamus):
        vec = [1.0, 0.0, 0.0]
        result = thalamus._cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self, thalamus):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = thalamus._cosine_similarity(vec1, vec2)
        assert abs(result) < 0.001

    def test_cosine_similarity_opposite(self, thalamus):
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        result = thalamus._cosine_similarity(vec1, vec2)
        assert abs(result + 1.0) < 0.001

    def test_cosine_similarity_empty(self, thalamus):
        assert thalamus._cosine_similarity([], []) == 0.0
        assert thalamus._cosine_similarity([1.0], []) == 0.0
        assert thalamus._cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity_different_lengths(self, thalamus):
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        result = thalamus._cosine_similarity(vec1, vec2)
        assert result == 0.0

    def test_get_summary_empty(self, thalamus):
        assert thalamus.get_summary() == ""

    def test_update_summary(self, thalamus):
        thalamus.update_summary("Test summary")
        assert thalamus.get_summary() == "Test summary"

    def test_get_memory_summary_empty(self, thalamus):
        assert thalamus.get_memory_summary() == ""

    def test_update_memory_summary(self, thalamus):
        thalamus.update_memory_summary("Memory test")
        assert thalamus.get_memory_summary() == "Memory test"

    def test_should_run_memory_analysis(self, thalamus):
        thalamus._metadata["last_analysis_count"] = 0
        assert thalamus.should_run_memory_analysis(5) == True
        assert thalamus.should_run_memory_analysis(4) == False

    def test_mark_memory_analysis_done(self, thalamus):
        thalamus.mark_memory_analysis_done(10)
        assert thalamus._metadata["last_analysis_count"] == 10

    def test_format_time_gap_none(self, thalamus):
        result = thalamus.format_time_gap(None)
        assert result == "First interaction"

    def test_format_time_gap_recent(self, thalamus):
        recent = datetime.now()
        result = thalamus.format_time_gap(recent)
        assert "Just now" in result or "hours" in result

    def test_get_last_interaction_none(self, thalamus):
        assert thalamus.get_last_interaction() is None

    def test_get_last_interaction_valid(self, thalamus):
        now = datetime.now()
        thalamus._metadata["last_interaction"] = now.isoformat()
        result = thalamus.get_last_interaction()
        assert result is not None
        assert result.date() == now.date()


class TestSessionMessage:

    def test_session_message_creation(self):
        msg = SessionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert msg.image_path is None
        assert msg.embedding is None

    def test_session_message_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        msg = SessionMessage(role="model", content="Hi there", embedding=embedding)
        assert msg.embedding == embedding


class TestContextPriority:

    def test_priority_ordering(self):
        assert ContextPriority.GLOBAL_CONTEXT.value > ContextPriority.PERSONA_CONTEXT.value
        assert ContextPriority.PERSONA_CONTEXT.value > ContextPriority.SCHEDULE_CONTEXT.value
        assert ContextPriority.SCHEDULE_CONTEXT.value > ContextPriority.RELEVANT_MEMORIES.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
