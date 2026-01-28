import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Deque, Optional, Set
from collections import defaultdict, deque

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60
ALLOWED_EXTENSIONS: Set[str] = {
    '.txt', '.md', '.py', '.json', '.csv', '.html',
    '.js', '.css', '.xml', '.yaml', '.yml', '.log', '.ini'
}

# User management state
USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LAST_ACTIVITY: Dict[str, datetime] = {}
RATE_LIMIT_TOKENS: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX))


async def get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific user."""
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]


def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    now = datetime.now()
    queue = RATE_LIMIT_TOKENS[user_id]

    while queue and queue[0] < now - timedelta(seconds=RATE_LIMIT_WINDOW):
        queue.popleft()

    if len(queue) < RATE_LIMIT_MAX:
        queue.append(now)
        return True
    return False


def update_activity(user_id: str) -> None:
    """Update last activity timestamp for user."""
    USER_LAST_ACTIVITY[user_id] = datetime.now()


def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


async def read_file_content(file_path: str) -> str:
    """Read file content with encoding fallback."""
    try:
        def _read_safe():
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return None

            with open(file_path, 'rb') as f:
                raw_data = f.read()

            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    return raw_data.decode(encoding)
                except UnicodeDecodeError:
                    continue

            return raw_data.decode('ascii', errors='ignore')

        content = await asyncio.to_thread(_read_safe)
        return content if content is not None else "[... File too large ...]"
    except Exception as e:
        logger.error(f"File read error: {e}")
        return "[Error reading file]"


async def send_chunked_response(update: Update, text: str) -> None:
    """Send long messages in chunks to avoid Telegram limits."""
    if not text:
        return

    MAX_LEN = 4000

    while text:
        if len(text) <= MAX_LEN:
            chunk = text
            text = ""
        else:
            split_idx = text.rfind('\n', 0, MAX_LEN)
            if split_idx == -1:
                split_idx = text.rfind(' ', 0, MAX_LEN)
            if split_idx == -1:
                split_idx = MAX_LEN

            chunk = text[:split_idx]
            text = text[split_idx:].lstrip()

        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            # Fallback to plain text if Markdown parsing fails
            await update.message.reply_text(chunk, parse_mode=None)

        if text:
            await asyncio.sleep(0.5)


async def handle_document(update: Update, user_dir: str, text_input: str) -> Optional[str]:
    """Handle document uploads."""
    import uuid
    doc = update.message.document
    fname = doc.file_name or "file.txt"
    ext = os.path.splitext(fname)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"❌ Format file tidak didukung\\. Hanya mendukung: {escape_markdown(', '.join(ALLOWED_EXTENSIONS))}",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return text_input if text_input else None

    if doc.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ File terlalu besar\\. Maksimal {MAX_FILE_SIZE // (1024*1024)} MB",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return text_input if text_input else None

    os.makedirs(user_dir, exist_ok=True)

    try:
        f_obj = await doc.get_file()
        tmp = os.path.join(user_dir, f"tmp_{uuid.uuid4().hex[:8]}{ext}")
        await f_obj.download_to_drive(tmp)

        content = await read_file_content(tmp)
        text_input += f"\n\n[FILE: {fname}]\n{content}\n[END FILE]"

        if os.path.exists(tmp):
            os.remove(tmp)

        return text_input
    except Exception as e:
        logger.error(f"Document handling error: {e}")
        await update.message.reply_text("❌ Gagal memproses file")
        return None


async def handle_photo(update: Update, user_dir: str) -> Optional[str]:
    """Handle photo uploads."""
    import uuid
    os.makedirs(user_dir, exist_ok=True)

    try:
        p_obj = await update.message.photo[-1].get_file()
        img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
        await p_obj.download_to_drive(img_path)
        return img_path
    except Exception as e:
        logger.error(f"Photo handling error: {e}")
        await update.message.reply_text("❌ Gagal memproses foto")
        return None
