import os
import asyncio
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import ContextTypes

from src.brain.brainstem import ADMIN_ID, NeuralEventBus
from src.brain.medulla_oblongata import handle_document, handle_photo

def get_brain_from_context(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get('brain')


def escape_markdown(text: str) -> str:
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def is_admin(user_id: int) -> bool:
    return str(user_id) == str(ADMIN_ID)


async def admin_only(update: Update) -> bool:
    if not is_admin(update.effective_user.id):
        await update.message.reply_text(
            "⛔ Access denied\\. This bot is for authorized users only\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return False
    return True


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    msg = (
        "👋 *Halo\\! Aku Vira\\.*\\n\\n"
        "Personal Life OS kamu\\. Aku bisa:\\n"
        "• Mengingat hal\\-hal penting\\n"
        "• Mengelola jadwal dan pengingat\\n"
        "• Berdiskusi tentang berbagai topik\\n\\n"
        "Ketik /help untuk melihat perintah\\."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    msg = (
        "🧠 *VIRA PERSONAL LIFE OS*\\n\\n"
        "*Perintah:*\\n"
        "/start \\- Mulai percakapan\\n"
        "/help \\- Tampilkan bantuan\\n"
        "/reset \\- Reset sesi chat\\n"
        "/status \\- Cek status sistem\\n"
        "/bio \\- Lihat/ubah info profil\\n\\n"
        "*Tips:*\\n"
        "• Kirim pesan untuk berbicara\\n"
        "• Kirim foto untuk dianalisis\\n"
        "• Minta pengingat waktu tertentu"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    brain = get_brain_from_context(context)
    if brain and brain.thalamus:
        brain.thalamus.clear_session()

    await update.message.reply_text(
        "🧹 Sesi chat telah di\\-reset\\. Mari mulai percakapan baru\\!",
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def cmd_instruction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    await update.message.reply_text(
        "ℹ️ Persona management tidak diperlukan di mode single\\-admin\\.",
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def cmd_bio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    brain = get_brain_from_context(context)
    args = context.args

    if not brain or not brain.hippocampus:
        await update.message.reply_text("⚠️ Sistem belum siap.", parse_mode=None)
        return

    if not args:
        profile = brain.hippocampus.admin_profile
        current_bio = profile.additional_info or "(Belum ada info)"
        current_bio = escape_markdown(current_bio)

        await update.message.reply_text(
            f"👤 *Info Profil:*\\n{current_bio}\\n\\n"
            f"Ketik `/bio [info baru]` untuk mengubah\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    new_info = " ".join(args)
    await brain.hippocampus.update_admin_profile(additional_info=new_info)
    new_info_escaped = escape_markdown(new_info)
    await update.message.reply_text(
        f"✅ Data tersimpan\\! Vira akan mengingat: \"{new_info_escaped}\"",
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return

    brain = get_brain_from_context(context)

    if not brain or not brain.hippocampus:
        await update.message.reply_text("⚠️ Sistem belum siap.")
        return

    m_stats = await brain.hippocampus.get_memory_stats()
    s_stats = await brain.hippocampus.get_schedule_stats()
    sys_stats = brain.prefrontal_cortex.get_system_stats() if brain.prefrontal_cortex else {}

    mood = brain.amygdala.mood.value if brain.amygdala else "unknown"
    satisfaction = brain.amygdala.satisfaction if brain.amygdala else 0.0

    msg = (
        "🧠 NEURAL SYSTEM STATUS\n\n"
        f"📡 API Health: {sys_stats.get('api_health', 'Unknown')}\n\n"
        f"📊 Memory Stats:\n"
        f"  • Active Memories: {m_stats.get('active', 0)}\n"
        f"  • Knowledge Triples: {m_stats.get('triples', 0)}\n"
        f"  • Pending Schedules: {s_stats.get('pending', 0)}\n\n"
        f"🎭 Emotional State:\n"
        f"  • Mood: {mood}\n"
        f"  • Satisfaction: {satisfaction:.2f}\n\n"
        f"🌐 Dashboard: http://localhost:5000"
    )

    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status")]]
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg,
        reply_markup=InlineKeyboardMarkup(kb)
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if not is_admin(query.from_user.id):
        return

    if query.data == "refresh_status":
        await query.delete_message()
        await cmd_status(update, context)


async def _keep_typing(bot, chat_id: int):
    """Keeps sending typing action every 4 seconds."""
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

    # Start persistent typing status
    typing_task = asyncio.create_task(_keep_typing(context.bot, update.effective_chat.id))

    brain = get_brain_from_context(context)

    if not brain or not brain.prefrontal_cortex:
        typing_task.cancel()
        await update.message.reply_text(
            "⚠️ Sistem belum siap. Restart bot dengan /start",
            parse_mode=None
        )
        return

    try:
        response = await brain.prefrontal_cortex.process(
            message=text_input,
            image_path=img_path,
            user_name=update.effective_user.first_name
        )

        typing_task.cancel() # Stop typing before sending response

        if response and response != "ERROR":
            await _send_chunked_response(update, response)
        else:
            print("❌ Processing failed silently (response was ERROR or None)")
            await update.message.reply_text(
                "⏳ Mohon tunggu sebentar, sistem sedang menyesuaikan diri...",
                parse_mode=None
            )
    except Exception as e:
        typing_task.cancel()
        import traceback
        traceback.print_exc()
        print(f"❌ Handler Error: {e}")
        await update.message.reply_text(
            "⏳ Mohon tunggu sebentar...",
            parse_mode=None
        )
    finally:
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass


async def _send_chunked_response(update: Update, text: str) -> None:
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
        except Exception:
            await update.message.reply_text(chunk, parse_mode=None)

        if text:
            await asyncio.sleep(0.5)

