from telegram import Update
from telegram.ext import ContextTypes

from src.brain.motor_cortex.context import is_admin
from src.brain.motor_cortex.commands import cmd_status


async def callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()

    if not is_admin(query.from_user.id):
        return

    if query.data == "refresh_status":
        await query.delete_message()
        await cmd_status(update, context)
