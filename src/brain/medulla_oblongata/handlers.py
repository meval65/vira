import logging
import os
import uuid
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode

from src.brain.medulla_oblongata.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from src.brain.medulla_oblongata.utils import escape_markdown, read_file_content

logger = logging.getLogger(__name__)


async def handle_document(
    update: Update, user_dir: str, text_input: str
) -> Optional[str]:
    doc = update.message.document
    fname = doc.file_name or "file.txt"
    ext = os.path.splitext(fname)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"❌ Format file tidak didukung\\. Hanya mendukung: {escape_markdown(', '.join(ALLOWED_EXTENSIONS))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return text_input if text_input else None

    if doc.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ File terlalu besar\\. Maksimal {MAX_FILE_SIZE // (1024*1024)} MB",
            parse_mode=ParseMode.MARKDOWN_V2,
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
        logger.error("Document handling error: %s", e)
        await update.message.reply_text("❌ Gagal memproses file")
        return None


async def handle_photo(update: Update, user_dir: str) -> Optional[str]:
    os.makedirs(user_dir, exist_ok=True)

    try:
        p_obj = await update.message.photo[-1].get_file()
        img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
        await p_obj.download_to_drive(img_path)
        return img_path
    except Exception as e:
        logger.error("Photo handling error: %s", e)
        await update.message.reply_text("❌ Gagal memproses foto")
        return None
