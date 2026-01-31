import logging

from src.brain.brainstem import ADMIN_ID, NeuralEventBus
from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return
    try:
        pending = await brain.hippocampus.get_pending_schedules(limit=10)
        for schedule in pending:
            schedule_id = schedule.get("id")
            schedule_context = schedule.get("context", "")
            if ADMIN_ID:
                try:
                    await NeuralEventBus.set_activity("cerebellum", "Sending Reminder")
                    if brain.prefrontal_cortex:
                        response = await brain.prefrontal_cortex.process(
                            message=f"[SCHEDULED REMINDER] {schedule_context}",
                            user_name="System",
                        )
                    else:
                        response = f"⏰ Pengingat: {schedule_context}"
                    await context.bot.send_message(
                        chat_id=int(ADMIN_ID),
                        text=response,
                    )
                    await brain.hippocampus.mark_schedule_executed(schedule_id, "delivered")
                    await NeuralEventBus.emit(
                        "cerebellum", "motor_cortex", "schedule_delivered",
                        payload={"schedule_id": schedule_id},
                    )
                except Exception as e:
                    logger.error("Failed to send schedule %s: %s", schedule_id, e)
                    await brain.hippocampus.mark_schedule_executed(
                        schedule_id, f"failed: {str(e)[:50]}",
                    )
                finally:
                    await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Schedule checker failed: %s", e)
