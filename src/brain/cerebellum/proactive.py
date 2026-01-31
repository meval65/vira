import logging

from telegram.constants import ParseMode

from src.brain.brainstem import ADMIN_ID, NeuralEventBus
from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_proactive_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.thalamus:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Proactive Check")
        message = await brain.thalamus.check_proactive_triggers()
        if message and ADMIN_ID:
            try:
                await context.bot.send_message(
                    chat_id=int(ADMIN_ID),
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                )
                await NeuralEventBus.emit(
                    "cerebellum", "motor_cortex", "proactive_message_sent",
                    payload={"message_length": len(message)},
                )
            except Exception as e:
                logger.error("Failed to send proactive message: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Proactive check failed: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")
