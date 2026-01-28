import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain.brainstem import (
    OpenRouterClient,
    ModelRotationConfig,
    ModelHealthScore,
    OPENROUTER_MODELS
)


class TestModelHealthScore:

    def test_initial_health_score(self):
        health = ModelHealthScore("test-model")
        assert health.health_score == 1.0
        assert health.success_count == 0
        assert health.failure_count == 0

    def test_record_success(self):
        health = ModelHealthScore("test-model")
        health.record_success(100.0)
        assert health.success_count == 1
        assert health.total_latency_ms == 100.0
        assert health.consecutive_failures == 0

    def test_record_failure(self):
        config = ModelRotationConfig()
        health = ModelHealthScore("test-model")
        health.record_failure(config)
        assert health.failure_count == 1
        assert health.consecutive_failures == 1

    def test_should_skip_after_max_failures(self):
        config = ModelRotationConfig(max_consecutive_failures=3)
        health = ModelHealthScore("test-model")
        for _ in range(3):
            health.record_failure(config)
        assert health.should_skip(config) == True

    def test_should_not_skip_before_max_failures(self):
        config = ModelRotationConfig(max_consecutive_failures=5)
        health = ModelHealthScore("test-model")
        health.record_failure(config)
        health.record_failure(config)
        assert health.should_skip(config) == False

    def test_consecutive_failures_reset_on_success(self):
        config = ModelRotationConfig(max_consecutive_failures=3)
        health = ModelHealthScore("test-model")
        health.record_failure(config)
        health.record_failure(config)
        health.record_success(100.0)
        assert health.consecutive_failures == 0


class TestModelRotationConfig:

    def test_default_values(self):
        config = ModelRotationConfig()
        assert config.retry_delay_base == 1.0
        assert config.retry_delay_max == 30.0
        assert config.health_recovery_minutes == 30
        assert config.max_consecutive_failures == 5
        assert config.tier_fallback_enabled == True


class TestOpenRouterClient:

    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="test-key")

    def test_get_tier_models(self, client):
        tier_1_models = client._get_tier_models("tier_1")
        tier_2_models = client._get_tier_models("tier_2")
        tier_3_models = client._get_tier_models("tier_3")
        assert isinstance(tier_1_models, list)
        assert isinstance(tier_2_models, list)
        assert isinstance(tier_3_models, list)

    def test_get_all_models_in_tier_order(self, client):
        all_models = client._get_all_models_in_tier_order()
        assert isinstance(all_models, list)
        assert len(all_models) > 0
        for tier, model in all_models:
            assert tier in ["tier_1", "tier_2", "tier_3"]
            assert isinstance(model, str)

    def test_select_best_model_with_tier(self, client):
        model = client._select_best_model(tier="tier_1")
        if OPENROUTER_MODELS.get("tier_1"):
            assert model is not None

    def test_get_health_score_creates_new(self, client):
        health = client._get_health_score("new-model")
        assert isinstance(health, ModelHealthScore)
        assert health.model_id == "new-model"

    def test_get_health_score_returns_existing(self, client):
        health1 = client._get_health_score("test-model")
        health1.record_success(100.0)
        health2 = client._get_health_score("test-model")
        assert health1 is health2
        assert health2.success_count == 1

    def test_get_status(self, client):
        status = client.get_status()
        assert "api_configured" in status
        assert "base_url" in status
        assert "health_scores" in status


class TestOpenRouterModels:

    def test_tier_structure(self):
        assert "tier_1" in OPENROUTER_MODELS
        assert "tier_2" in OPENROUTER_MODELS
        assert "tier_3" in OPENROUTER_MODELS

    def test_tiers_have_models(self):
        for tier, models in OPENROUTER_MODELS.items():
            assert isinstance(models, list)
            assert len(models) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
