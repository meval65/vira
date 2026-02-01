from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.brain.brainstem import ADMIN_ID


def get_brain_from_context(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get("brain")


def escape_markdown(text: str) -> str:
    special_chars = [
        "_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


def is_admin(user_id: int) -> bool:
    return str(user_id) == str(ADMIN_ID)


async def admin_only(update: Update) -> bool:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text(
            "⛔ Access denied\\. This bot is for authorized users only\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return False
    return True


