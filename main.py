import os
import logging
import asyncio
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, 
    MessageHandler, CallbackQueryHandler, filters, Application
)
from telegram.error import TelegramError, RetryAfter, TimedOut

from src.database import DBConnection
from src.services.memory_service import MemoryManager
from src.services.analyzer_service import MemoryAnalyzer
from src.services.scheduler_service import SchedulerService
from src.services.chat_service import ChatHandler
from src.config import get_available_chat_models, CHAT_MODEL

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

db = None
mem_mgr = None
chat_handler = None
scheduler = None

USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LAST_ACTIVITY: Dict[str, datetime] = {}
RATE_LIMIT_CACHE: Dict[str, list] = defaultdict(list)

MAX_FILE_SIZE = 5 * 1024 * 1024
RATE_LIMIT_MESSAGES = 10
RATE_LIMIT_WINDOW = 60
ALLOWED_TEXT_EXTENSIONS = ('.txt', '.md', '.py', '.json', '.csv', '.html', '.js', '.css', '.xml', '.yaml', '.yml', '.log')

async def get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]

def check_rate_limit(user_id: str) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    RATE_LIMIT_CACHE[user_id] = [
        ts for ts in RATE_LIMIT_CACHE[user_id] if ts > cutoff
    ]
    
    if len(RATE_LIMIT_CACHE[user_id]) >= RATE_LIMIT_MESSAGES:
        return False
    
    RATE_LIMIT_CACHE[user_id].append(now)
    return True

def update_user_activity(user_id: str):
    USER_LAST_ACTIVITY[user_id] = datetime.now()

async def cleanup_inactive_locks():
    now = datetime.now()
    timeout = timedelta(minutes=30)
    
    to_remove = [
        uid for uid, last_active in USER_LAST_ACTIVITY.items()
        if now - last_active > timeout and uid in USER_LOCKS and not USER_LOCKS[uid].locked()
    ]
    
    for uid in to_remove:
        USER_LOCKS.pop(uid, None)
        USER_LAST_ACTIVITY.pop(uid, None)
        RATE_LIMIT_CACHE.pop(uid, None)
    
    if to_remove:
        logger.info(f"[CLEANUP] Removed {len(to_remove)} inactive user sessions")

async def scheduled_maintenance(context: ContextTypes.DEFAULT_TYPE):
    logger.info("[MAINTENANCE] Starting scheduled cleanup...")
    
    try:
        await asyncio.to_thread(mem_mgr.apply_decay_rules)
        await asyncio.to_thread(mem_mgr.optimize_memories, user_id=None)
        
        cleaned = await asyncio.to_thread(scheduler.cleanup_old_schedules, days_old=30)
        logger.info(f"[MAINTENANCE] Cleaned {cleaned} old schedules")
        
        await cleanup_inactive_locks
        
        logger.info("[MAINTENANCE] Complete")
    except Exception as e:
        logger.error(f"[MAINTENANCE] Failed: {e}", exc_info=True)

async def background_scheduler_checker(context: ContextTypes.DEFAULT_TYPE):
    try:
        due_schedules = await asyncio.to_thread(scheduler.get_due_schedules, lookback_minutes=5)
        
        if not due_schedules:
            return

        for item in due_schedules:
            user_id = item['user_id']
            schedule_id = item['id']
            ctx = item['context']
            priority = item.get('priority', 0)
            
            try:
                logger.info(f"[SCHEDULER] Processing schedule {schedule_id} for user {user_id}")
                
                ai_response = await asyncio.to_thread(
                    chat_handler.trigger_proactive_message,
                    user_id,
                    ctx
                )
                
                if ai_response:
                    priority_marker = "⚠️ " if priority > 0 else ""
                    message_text = f"{priority_marker}{ai_response}"
                    
                    try:
                        await context.bot.send_message(
                            chat_id=int(user_id),
                            text=message_text,
                            parse_mode=ParseMode.HTML if '<' in message_text else None
                        )
                        
                        await asyncio.to_thread(
                            scheduler.mark_as_executed, 
                            schedule_id, 
                            "Sent successfully"
                        )
                        
                        logger.info(f"[SCHEDULER] Successfully sent schedule {schedule_id}")
                        
                    except (TelegramError, RetryAfter, TimedOut) as e:
                        logger.warning(f"[SCHEDULER] Failed to send to {user_id}: {e}")
                        await asyncio.to_thread(
                            scheduler.mark_as_executed,
                            schedule_id,
                            f"Failed: {str(e)}"
                        )
                        
            except Exception as e:
                logger.error(f"[SCHEDULER] Error processing {schedule_id}: {e}", exc_info=True)
                
    except Exception as e:
        logger.error(f"[SCHEDULER] Checker error: {e}", exc_info=True)

async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    chat_m = chat_handler.chat_model_name
    anal_m = chat_handler.analysis_model_name
    l1_m = chat_handler.first_level_model_name
    
    mem_stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
    sched_stats = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
    sys_stats = chat_handler.get_system_stats()
    
    status_text = (
        "🤖 <b>VIRA SYSTEM STATUS</b>\n\n"
        "<b>📊 AI Models:</b>\n"
        f"├ Chat: <code>{chat_m}</code>\n"
        f"├ L1 Analyzer: <code>{l1_m}</code>\n"
        f"└ L2 Extractor: <code>{anal_m}</code>\n\n"
        "<b>🧠 Memory Bank:</b>\n"
        f"├ Active: {mem_stats.get('active', 0)}\n"
        f"├ Archived: {mem_stats.get('archived', 0)}\n"
        f"├ Avg Priority: {mem_stats.get('avg_priority', 0)}\n"
        f"└ Avg Usage: {mem_stats.get('avg_use_count', 0)}\n\n"
        "<b>📅 Scheduler:</b>\n"
        f"├ Pending: {sched_stats.get('pending', 0)}\n"
        f"├ Executed: {sched_stats.get('executed', 0)}\n"
        f"└ Cancelled: {sched_stats.get('cancelled', 0)}\n\n"
        "<b>⚙️ System:</b>\n"
        f"├ Active Sessions: {sys_stats.get('active_sessions', 0)}\n"
        f"├ API Key: #{sys_stats.get('current_api_key_index', 0)}\n"
        f"└ Status: ✅ Operational"
    )
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.HTML)

async def handle_check_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    try:
        history = await asyncio.to_thread(
            scheduler.get_schedule_history,
            user_id,
            limit=15,
            include_pending=True
        )
        
        if not history:
            await update.message.reply_text(
                "📅 <b>Tidak Ada Jadwal</b>\n\nBelum ada pengingat yang terdaftar.",
                parse_mode=ParseMode.HTML
            )
            return

        pending = [s for s in history if s['status'] == 'pending']
        executed = [s for s in history if s['status'] == 'executed']
        
        msg = "🗓️ <b>DAFTAR PENGINGAT</b>\n\n"
        
        if pending:
            msg += "⏳ <b>AKTIF:</b>\n"
            for item in pending[:10]:
                waktu = item['scheduled_at']
                if isinstance(waktu, str):
                    try:
                        dt = datetime.fromisoformat(waktu)
                        waktu = dt.strftime("%d/%m %H:%M")
                    except:
                        pass
                
                msg += f"• <b>{waktu}</b>\n  <i>{item['context'][:60]}...</i>\n"
            msg += "\n"
        
        if executed:
            msg += "✅ <b>TERAKHIR DIEKSEKUSI:</b>\n"
            for item in executed[:3]:
                waktu = item['scheduled_at']
                if isinstance(waktu, str):
                    try:
                        dt = datetime.fromisoformat(waktu)
                        waktu = dt.strftime("%d/%m %H:%M")
                    except:
                        pass
                msg += f"• {waktu}: {item['context'][:40]}...\n"
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_schedule")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            msg,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"[SCHEDULE-CHECK] Error: {e}", exc_info=True)
        await update.message.reply_text("❌ Gagal mengambil data jadwal.")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = str(query.from_user.id)
    
    try:
        if query.data == "refresh_schedule":
            history = await asyncio.to_thread(
                scheduler.get_schedule_history,
                user_id,
                limit=15,
                include_pending=True
            )
            
            pending = [s for s in history if s['status'] == 'pending']
            
            msg = f"🗓️ <b>JADWAL DIPERBARUI</b>\n\n⏳ Aktif: {len(pending)}\n⏰ Terakhir update: {datetime.now().strftime('%H:%M:%S')}"
            
            if pending:
                msg += "\n\n"
                for item in pending[:5]:
                    waktu = item['scheduled_at']
                    if isinstance(waktu, str):
                        try:
                            dt = datetime.fromisoformat(waktu)
                            waktu = dt.strftime("%d/%m %H:%M")
                        except:
                            pass
                    msg += f"• {waktu}: {item['context'][:40]}...\n"
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_schedule")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                msg,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
        elif query.data == "refresh_memory":
            cursor = db.get_cursor()
            cursor.execute(
                """SELECT memory_type, summary, use_count, priority 
                FROM memories WHERE user_id=? AND status='active' 
                ORDER BY last_used_at DESC LIMIT 12""",
                (user_id,)
            )
            memories = cursor.fetchall()
            
            if not memories:
                await query.edit_message_text("🧠 Otak saya masih kosong tentangmu.")
                return

            stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
            
            response = (
                f"🧠 <b>MEMORY BANK</b>\n\n"
                f"📊 Total: {stats['active']} aktif | {stats['archived']} arsip\n"
                f"⭐ Avg Priority: {stats['avg_priority']}\n"
                f"⏰ Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
                f"<b>12 MEMORI TERATAS:</b>\n\n"
            )
            
            icons = {"emotion": "❤️", "preference": "⭐", "decision": "🔨", "boundary": "🛡️"}
            
            for row in memories:
                m_type, summary, count, priority = row
                icon = icons.get(m_type, "📌")
                response += (
                    f"{icon} <b>[{m_type.upper()}]</b> "
                    f"({count}x, P:{priority:.1f})\n"
                    f"└ <i>{summary[:70]}...</i>\n\n"
                )
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_memory")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                response,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
        elif query.data == "refresh_stats":
            mem_stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
            sched_stats = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
            db_stats = await asyncio.to_thread(db.get_database_stats)
            
            next_schedule = ""
            if sched_stats.get('next_schedule'):
                next_time = sched_stats['next_schedule']['time']
                next_ctx = sched_stats['next_schedule']['context']
                if isinstance(next_time, str):
                    try:
                        dt = datetime.fromisoformat(next_time)
                        next_time = dt.strftime("%d/%m %H:%M")
                    except:
                        pass
                next_schedule = f"\n└ Next: {next_time} - {next_ctx[:30]}..."
            
            stats_text = (
                "📊 <b>STATISTIK DETAIL</b>\n\n"
                "<b>🧠 Memory Bank (Personal):</b>\n"
                f"├ Active: {mem_stats.get('active', 0)}\n"
                f"├ Archived: {mem_stats.get('archived', 0)}\n"
                f"├ Total: {mem_stats.get('total', 0)}\n"
                f"├ Avg Priority: {mem_stats.get('avg_priority', 0)}\n"
                f"├ Avg Usage: {mem_stats.get('avg_use_count', 0)}\n"
                f"└ Total Retrievals: {mem_stats.get('total_retrievals', 0)}\n\n"
                "<b>📅 Scheduler (Personal):</b>\n"
                f"├ Pending: {sched_stats.get('pending', 0)}\n"
                f"├ Executed: {sched_stats.get('executed', 0)}\n"
                f"├ Cancelled: {sched_stats.get('cancelled', 0)}\n"
                f"└ Total: {sched_stats.get('total', 0)}{next_schedule}\n\n"
                "<b>💾 Database (Global):</b>\n"
                f"├ Total Users: {db_stats.get('total_users', 0)}\n"
                f"├ Active Memories: {db_stats.get('active_memories', 0)}\n"
                f"├ Pending Schedules: {db_stats.get('pending_schedules', 0)}\n"
                f"└ DB Size: {db_stats.get('db_size_mb', 0)} MB\n\n"
                f"<i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>"
            )
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                stats_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"[CALLBACK] Error: {e}", exc_info=True)
        try:
            await query.edit_message_text("❌ Gagal memperbarui data.")
        except:
            pass

async def handle_forget_everything(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    confirm_msg = await update.message.reply_text(
        "💥 <i>Menghapus semua data...</i>",
        parse_mode=ParseMode.HTML
    )
    
    try:
        deleted_mem = await asyncio.to_thread(mem_mgr.wipe_all_memories, user_id)
        
        await asyncio.to_thread(chat_handler.clear_session, user_id)
        
        cursor = db.get_cursor()
        cursor.execute(
            "UPDATE schedules SET status = 'cancelled' WHERE user_id = ? AND status = 'pending'",
            (user_id,)
        )
        cancelled_sched = cursor.rowcount
        db.commit()
        
        user_dir = os.path.join("storage", "userdata", user_id)
        if os.path.exists(user_dir):
            await asyncio.to_thread(shutil.rmtree, user_dir, ignore_errors=True)

        await confirm_msg.edit_text(
            f"✅ <b>HARD RESET SELESAI</b>\n\n"
            f"📊 Dihapus:\n"
            f"├ {deleted_mem} memori\n"
            f"├ {cancelled_sched} jadwal\n"
            f"└ Semua file & sesi\n\n"
            f"Kita mulai dari awal lagi! 🌟",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"[FORGET] Error: {e}", exc_info=True)
        await confirm_msg.edit_text("❌ Terjadi kesalahan saat mereset data.")

async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    await asyncio.to_thread(chat_handler.clear_session, user_id)
    
    await update.message.reply_text(
        "🧹 <b>Sesi Baru Dimulai!</b>\n\n"
        "Konteks percakapan telah dihapus.\n"
        "Tapi ingatan penting tentangmu tetap tersimpan.",
        parse_mode=ParseMode.HTML
    )

async def handle_check_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    status_msg = await update.message.reply_text(
        "🔍 <i>Mengakses Neural Bank...</i>",
        parse_mode=ParseMode.HTML
    )

    try:
        if context.args and context.args[0]:
            keyword = context.args[0]
            results = await asyncio.to_thread(
                mem_mgr.search_memories,
                user_id,
                keyword,
                limit=10
            )
            
            if not results:
                await status_msg.edit_text(
                    f"🔍 Tidak ada memori yang cocok dengan '<b>{keyword}</b>'",
                    parse_mode=ParseMode.HTML
                )
                return
            
            response = f"🔎 <b>HASIL PENCARIAN: {keyword}</b>\n\n"
            icons = {"emotion": "❤️", "preference": "⭐", "decision": "🔨", "boundary": "🛡️"}
            
            for mem in results:
                icon = icons.get(mem['type'], "📌")
                response += (
                    f"{icon} <b>[{mem['type'].upper()}]</b> "
                    f"(P:{mem['priority']:.1f}, U:{mem['use_count']})\n"
                    f"└ <i>{mem['summary'][:80]}...</i>\n\n"
                )
        else:
            cursor = db.get_cursor()
            cursor.execute(
                """SELECT memory_type, summary, use_count, priority 
                FROM memories WHERE user_id=? AND status='active' 
                ORDER BY last_used_at DESC LIMIT 12""",
                (user_id,)
            )
            memories = cursor.fetchall()
            
            if not memories:
                await status_msg.edit_text("🧠 Otak saya masih kosong tentangmu.")
                return

            stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
            
            response = (
                f"🧠 <b>MEMORY BANK</b>\n\n"
                f"📊 Total: {stats['active']} aktif | {stats['archived']} arsip\n"
                f"⭐ Avg Priority: {stats['avg_priority']}\n\n"
                f"<b>12 MEMORI TERATAS:</b>\n\n"
            )
            
            icons = {"emotion": "❤️", "preference": "⭐", "decision": "🔨", "boundary": "🛡️"}
            
            for row in memories:
                m_type, summary, count, priority = row
                icon = icons.get(m_type, "📌")
                response += (
                    f"{icon} <b>[{m_type.upper()}]</b> "
                    f"({count}x, P:{priority:.1f})\n"
                    f"└ <i>{summary[:70]}...</i>\n\n"
                )
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_memory")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await status_msg.edit_text(
            response,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"[MEMORY-CHECK] Error: {e}", exc_info=True)
        await status_msg.edit_text("❌ Gagal mengambil data memori.")

async def handle_change_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    available_models = get_available_chat_models()
    
    if not context.args:
        current = chat_handler.chat_model_name
        list_str = "\n".join(f"• <code>{m}</code>" for m in available_models)
        await update.message.reply_text(
            f"🔋 <b>Model Aktif:</b> <code>{current}</code>\n\n"
            f"🔄 <b>Tersedia:</b>\n{list_str}\n\n"
            f"Cara pakai: <code>/model nama_model</code>",
            parse_mode=ParseMode.HTML
        )
        return
    
    new_model = context.args[0]
    if new_model not in available_models:
        await update.message.reply_text(
            "❌ Model tidak valid.\n\nGunakan <code>/model</code> untuk melihat daftar.",
            parse_mode=ParseMode.HTML
        )
        return
    
    try:
        await asyncio.to_thread(chat_handler.change_model, new_model)
        await update.message.reply_text(
            f"✅ Model berhasil diganti ke:\n<code>{new_model}</code>",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"[MODEL-CHANGE] Error: {e}")
        await update.message.reply_text(f"❌ Gagal mengganti model: {str(e)[:100]}")

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 <b>VIRA - Personal AI Assistant</b>\n\n"
        "<b>📋 Perintah Tersedia:</b>\n\n"
        "🔹 <code>/start</code> - Mulai bot\n"
        "🔹 <code>/help</code> - Bantuan ini\n"
        "🔹 <code>/status</code> - Cek status sistem\n"
        "🔹 <code>/memory [keyword]</code> - Lihat/cari memori\n"
        "🔹 <code>/jadwal</code> - Lihat daftar pengingat\n"
        "🔹 <code>/model [nama]</code> - Ganti AI model\n"
        "🔹 <code>/reset</code> - Reset sesi chat\n"
        "🔹 <code>/forget</code> - Hapus semua data\n"
        "🔹 <code>/stats</code> - Statistik detail\n\n"
        "<b>💡 Fitur Otomatis:</b>\n"
        "• Memori jangka panjang dengan AI\n"
        "• Pengingat proaktif otomatis\n"
        "• Analisis konteks mendalam\n"
        "• Multi-modal (teks, gambar, file)\n"
        "• Smart scheduling & reminders\n\n"
        "<b>🎯 Contoh Penggunaan:</b>\n"
        "• <i>\"Besok jam 8 ingetin aku meeting\"</i>\n"
        "• <i>\"Aku suka kopi tanpa gula\"</i>\n"
        "• <i>\"Apa yang pernah kuceritakan tentang hobi?\"</i>\n\n"
        "Kirim pesan apa saja untuk memulai! 🚀"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_user_activity(user_id)
    
    status_msg = await update.message.reply_text(
        "📊 <i>Mengumpulkan statistik...</i>",
        parse_mode=ParseMode.HTML
    )
    
    try:
        mem_stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
        sched_stats = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
        db_stats = await asyncio.to_thread(db.get_database_stats)
        
        next_schedule = ""
        if sched_stats.get('next_schedule'):
            next_time = sched_stats['next_schedule']['time']
            next_ctx = sched_stats['next_schedule']['context']
            if isinstance(next_time, str):
                try:
                    dt = datetime.fromisoformat(next_time)
                    next_time = dt.strftime("%d/%m %H:%M")
                except:
                    pass
            next_schedule = f"\n└ Next: {next_time} - {next_ctx[:30]}..."
        
        stats_text = (
            "📊 <b>STATISTIK DETAIL</b>\n\n"
            "<b>🧠 Memory Bank (Personal):</b>\n"
            f"├ Active: {mem_stats.get('active', 0)}\n"
            f"├ Archived: {mem_stats.get('archived', 0)}\n"
            f"├ Total: {mem_stats.get('total', 0)}\n"
            f"├ Avg Priority: {mem_stats.get('avg_priority', 0)}\n"
            f"├ Avg Usage: {mem_stats.get('avg_use_count', 0)}\n"
            f"└ Total Retrievals: {mem_stats.get('total_retrievals', 0)}\n\n"
            "<b>📅 Scheduler (Personal):</b>\n"
            f"├ Pending: {sched_stats.get('pending', 0)}\n"
            f"├ Executed: {sched_stats.get('executed', 0)}\n"
            f"├ Cancelled: {sched_stats.get('cancelled', 0)}\n"
            f"└ Total: {sched_stats.get('total', 0)}{next_schedule}\n\n"
            "<b>💾 Database (Global):</b>\n"
            f"├ Total Users: {db_stats.get('total_users', 0)}\n"
            f"├ Active Memories: {db_stats.get('active_memories', 0)}\n"
            f"├ Pending Schedules: {db_stats.get('pending_schedules', 0)}\n"
            f"└ DB Size: {db_stats.get('db_size_mb', 0)} MB\n\n"
            f"<i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>"
        )
        
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await status_msg.edit_text(
            stats_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"[STATS] Error: {e}", exc_info=True)
        await status_msg.edit_text("❌ Gagal mengambil statistik.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id
    
    if not check_rate_limit(user_id):
        await update.message.reply_text(
            "⏳ <b>Rate Limit</b>\n\n"
            f"Terlalu banyak pesan. Tunggu {RATE_LIMIT_WINDOW} detik.",
            parse_mode=ParseMode.HTML
        )
        return
    
    update_user_activity(user_id)
    
    user_text = update.message.text or update.message.caption or ""
    image_path = None
    
    user_storage_dir = os.path.join("storage", "userdata", user_id)
    if not os.path.exists(user_storage_dir):
        os.makedirs(user_storage_dir)

    if update.message.document:
        doc = update.message.document
        file_name = doc.file_name or "document.txt"
        file_size = doc.file_size or 0
        
        if file_name.lower().endswith(ALLOWED_TEXT_EXTENSIONS):
            if file_size > MAX_FILE_SIZE:
                await update.message.reply_text(
                    f"⚠️ <b>File Terlalu Besar</b>\n\n"
                    f"Maksimal {MAX_FILE_SIZE // (1024*1024)}MB",
                    parse_mode=ParseMode.HTML
                )
                return
            
            try:
                status_msg = await update.message.reply_text(
                    f"📂 <i>Membaca {file_name}...</i>",
                    parse_mode=ParseMode.HTML
                )
                
                doc_file = await doc.get_file()
                file_content_bytes = await doc_file.download_as_bytearray()
                
                try:
                    file_content_str = file_content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        file_content_str = file_content_bytes.decode('latin-1')
                    except:
                        await status_msg.edit_text("❌ File encoding tidak didukung.")
                        return
                
                if len(file_content_str) > 50000:
                    file_content_str = file_content_str[:50000] + "\n\n[... file dipotong ...]"
                
                user_text += f"\n\n--- [FILE: {file_name}] ---\n{file_content_str}\n--- [END] ---"
                
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"[FILE] Read error: {e}")
                await update.message.reply_text("❌ Gagal membaca file.")
                return
        else:
            if not user_text:
                await update.message.reply_text(
                    "⚠️ <b>Format Tidak Didukung</b>\n\n"
                    "Saya hanya bisa membaca file teks.\n"
                    f"Format didukung: {', '.join(ALLOWED_TEXT_EXTENSIONS[:5])}...",
                    parse_mode=ParseMode.HTML
                )
                return

    if update.message.photo:
        try:
            photo_file = await update.message.photo[-1].get_file()
            filename = f"img_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(user_storage_dir, filename)
            await photo_file.download_to_drive(image_path)
        except Exception as e:
            logger.error(f"[IMAGE] Download error: {e}")
            await update.message.reply_text("❌ Gagal mengunduh gambar.")
            return

    if not user_text and not image_path:
        return

    lock = await get_user_lock(user_id)
    
    if lock.locked():
        await update.message.reply_text(
            "⏳ <i>Mohon tunggu, pesan sebelumnya sedang diproses...</i>",
            parse_mode=ParseMode.HTML
        )
        return

    async with lock:
        typing_task = asyncio.create_task(
            periodic_typing(context.bot, chat_id, duration=30)
        )
        
        try:
            response_text = await asyncio.to_thread(
                chat_handler.process_message,
                user_id,
                user_text,
                image_path
            )
            
            typing_task.cancel()
            
            if not response_text or response_text == "ERROR":
                response_text = "😵 Maaf, terjadi kesalahan internal. Coba lagi?"
            
            if len(response_text) > 4096:
                chunks = [response_text[i:i+4096] for i in range(0, len(response_text), 4096)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
                    await asyncio.sleep(0.5)
            else:
                await update.message.reply_text(response_text)
            
        except Exception as e:
            typing_task.cancel()
            logger.error(f"[PROCESS] Error: {e}", exc_info=True)
            
            error_msg = "😵 <b>System Error</b>\n\nTerjadi gangguan pada sistem."
            
            if "quota" in str(e).lower():
                error_msg = "⚠️ <b>API Quota Exceeded</b>\n\nSemua API key sedang overload. Coba beberapa saat lagi."
            elif "timeout" in str(e).lower():
                error_msg = "⏱️ <b>Timeout</b>\n\nProses terlalu lama. Coba pesan lebih pendek."
            
            await update.message.reply_text(error_msg, parse_mode=ParseMode.HTML)

async def periodic_typing(bot, chat_id: int, duration: int):
    try:
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[TYPING] Error: {e}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"[ERROR-HANDLER] Update {update} caused error: {context.error}", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ <b>Unexpected Error</b>\n\nTerjadi kesalahan yang tidak terduga.",
                parse_mode=ParseMode.HTML
            )
        except:
            pass

async def post_init(application: Application):
    global db, mem_mgr, chat_handler, scheduler
    
    print("═" * 50)
    print("[INIT] Starting Vira AI System...")
    
    db = DBConnection()
    mem_mgr = MemoryManager(db)
    analyzer = MemoryAnalyzer()
    scheduler = SchedulerService(db)
    chat_handler = ChatHandler(mem_mgr, analyzer, scheduler)
    
    if application.job_queue:
        application.job_queue.run_repeating(
            scheduled_maintenance,
            interval=86400,
            first=300
        )
        
        application.job_queue.run_repeating(
            background_scheduler_checker,
            interval=30,
            first=15
        )
        
        application.job_queue.run_repeating(
            cleanup_inactive_locks,
            interval=1800,
            first=600
        )
        
        print("[INIT] Background jobs scheduled:")
        print("  • Maintenance: 24h interval")
        print("  • Scheduler: 30s interval")
        print("  • Cleanup: 30m interval")

    print(f"[INIT] Active chat model: {CHAT_MODEL}")
    print("═" * 50)

async def post_shutdown(application: Application):
    print("\n[SHUTDOWN] Cleaning up resources...")
    
    if db:
        db.close()
    
    print("[SHUTDOWN] Complete. Goodbye! 👋")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
        exit(1)

    try:
        app = (
            ApplicationBuilder()
            .token(TOKEN)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .concurrent_updates(True)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .pool_timeout(30.0)
            .build()
        )
    except Exception as e:
        logger.error(f"[INIT] Failed to build application: {e}")
        exit(1)
    
    app.add_handler(CommandHandler('start', handle_help))
    app.add_handler(CommandHandler('help', handle_help))
    app.add_handler(CommandHandler('newchat', handle_reset))
    app.add_handler(CommandHandler('reset', handle_reset))
    app.add_handler(CommandHandler('memory', handle_check_memory))
    app.add_handler(CommandHandler('model', handle_change_model))
    app.add_handler(CommandHandler('forget', handle_forget_everything))
    app.add_handler(CommandHandler('status', handle_status))
    app.add_handler(CommandHandler('stats', handle_stats))
    app.add_handler(CommandHandler('jadwal', handle_check_schedule))
    app.add_handler(CommandHandler('schedule', handle_check_schedule))
    
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.Document.ALL) & (~filters.COMMAND),
        handle_message
    ))
    
    app.add_error_handler(error_handler)
    
    print("🤖 Vira Bot is now running...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            close_loop=False
        )
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Received interrupt signal")
    except Exception as e:
        logger.error(f"[RUNTIME] Critical error: {e}", exc_info=True)
    finally:
        print("[SHUTDOWN] Bot stopped")