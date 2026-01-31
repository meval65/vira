import asyncio
import logging
import os

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError

from src.brain.medulla_oblongata.constants import MAX_FILE_SIZE

logger = logging.getLogger(__name__)


def escape_markdown(text: str) -> str:
    special_chars = [
        "_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!",
    ]
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


async def read_file_content(file_path: str) -> str:
    try:
        def _read_safe():
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return None
            with open(file_path, "rb") as f:
                raw_data = f.read()
            for encoding in ["utf-8", "latin-1", "ascii"]:
                try:
                    return raw_data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw_data.decode("ascii", errors="ignore")

        content = await asyncio.to_thread(_read_safe)
        return content if content is not None else "[... File too large ...]"
    except Exception as e:
        logger.error("File read error: %s", e)
        return "[Error reading file]"


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
        except TelegramError:
            await update.message.reply_text(chunk, parse_mode=None)
        if text:
            await asyncio.sleep(0.5)
