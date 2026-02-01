import asyncio
import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.brain.medulla_oblongata import handle_document, handle_photo
from src.brain.motor_cortex.context import get_brain_from_context, is_admin
from src.brain.motor_cortex.utils import send_chunked_response

logger = logging.getLogger(__name__)


async def _keep_typing(bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update.effective_user.id):
        return

    text_input = update.message.text or update.message.caption or ""
    img_path = None
    user_dir = os.path.join("storage", "userdata", "admin")

    if update.message.document:
        text_input = await handle_document(update, user_dir, text_input)
        if text_input is None:
            return

    if update.message.photo:
        img_path = await handle_photo(update, user_dir)
        if img_path is None and not text_input:
            return

    if not text_input and not img_path:
        return

    typing_task = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id)
    )

    brain = get_brain_from_context(context)

    if not brain or not brain.prefrontal_cortex:
        typing_task.cancel()
        await update.message.reply_text(
            "⚠️ Sistem belum siap. Restart bot dengan /start",
            parse_mode=None,
        )
        return

    try:
        response = await brain.prefrontal_cortex.process(
            message=text_input,
            image_path=img_path,
            user_name=update.effective_user.first_name,
        )

        typing_task.cancel()

        if response and response != "ERROR":
            await send_chunked_response(update, response)
        else:
            logger.error("Processing failed silently (response was ERROR or None)")
            await update.message.reply_text(
                "⏳ Mohon tunggu sebentar, sistem sedang menyesuaikan diri...",
                parse_mode=None,
            )
    except Exception as e:
        typing_task.cancel()
        logger.exception("Handler error: %s", e)
        await update.message.reply_text(
            "⏳ Mohon tunggu sebentar...",
            parse_mode=None,
        )
    finally:
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass


