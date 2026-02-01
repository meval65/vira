import logging

from src.brain.brainstem import NeuralEventBus
from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_emotional_decay(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.amygdala:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Emotional Decay")
        current = brain.amygdala.satisfaction
        if abs(current) > 0.1:
            decay_amount = 0.05 if current > 0 else -0.05
            brain.amygdala._state.satisfaction_level = max(
                -1, min(1, current - decay_amount)
            )
        await brain.amygdala.save_state()
        await NeuralEventBus.emit(
            "cerebellum", "amygdala", "emotional_decay",
            payload={"new_satisfaction": brain.amygdala.satisfaction},
        )
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Emotional decay failed: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")


