import logging

from src.brain.brainstem import NeuralEventBus
from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_session_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.thalamus:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Session Cleanup")
        await brain.thalamus.cleanup_session()
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Session cleanup failed: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")


