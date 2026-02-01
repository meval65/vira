from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.brain.motor_cortex.context import (
    get_brain_from_context,
    escape_markdown,
    admin_only,
)


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
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_instruction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await admin_only(update):
        return
    await update.message.reply_text(
        "ℹ️ Persona management tidak diperlukan di mode single\\-admin\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
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
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    new_info = " ".join(args)
    await brain.hippocampus.update_admin_profile(additional_info=new_info)
    new_info_escaped = escape_markdown(new_info)
    await update.message.reply_text(
        f"✅ Data tersimpan\\! Vira akan mengingat: \"{new_info_escaped}\"",
        parse_mode=ParseMode.MARKDOWN_V2,
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
    sys_stats = (
        brain.prefrontal_cortex.get_system_stats()
        if brain.prefrontal_cortex
        else {}
    )

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
        reply_markup=InlineKeyboardMarkup(kb),
    )


