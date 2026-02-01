import logging
from datetime import datetime

from src.brain.brainstem import NeuralEventBus
from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_health_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain:
        return
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "hippocampus": brain.hippocampus is not None,
            "prefrontal_cortex": brain.prefrontal_cortex is not None,
            "amygdala": brain.amygdala is not None,
            "thalamus": brain.thalamus is not None,
            "openrouter": brain.openrouter is not None,
        }
        if brain.openrouter:
            api_status = brain.openrouter.get_status()
            health_status["api_configured"] = api_status.get("api_configured", False)
            health_status["failed_models"] = len(api_status.get("failed_models", []))
        await NeuralEventBus.emit(
            "cerebellum", "dashboard", "health_check",
            payload=health_status,
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)


