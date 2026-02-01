import asyncio

from telegram import Update
from telegram.constants import ParseMode


async def send_chunked_response(update: Update, text: str) -> None:
    if not text:
        return
    MAX_LEN = 4000
    while text:
        if len(text) <= MAX_LEN:
            chunk = text
            text = ""
        else:
            split_idx = text.rfind("\n", 0, MAX_LEN)
            if split_idx == -1:
                split_idx = text.rfind(" ", 0, MAX_LEN)
            if split_idx == -1:
                split_idx = MAX_LEN
            chunk = text[:split_idx]
            text = text[split_idx:].lstrip()
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk, parse_mode=None)
        if text:
            await asyncio.sleep(0.5)


