import os
import logging
import asyncio
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, Deque, Optional, Set, List
from collections import defaultdict, deque
from dotenv import load_dotenv
import json
import tempfile

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, CallbackQueryHandler, filters, Application, Defaults
)
from telegram.error import TelegramError

from src.database import DBConnection
from src.services.memory_manager import MemoryManager
from src.services.embedding import MemoryAnalyzer
from src.services.scheduler_service import SchedulerService
from src.services.chat_manager import ChatHandler
from src.config import get_available_chat_models, CHAT_MODEL, EMBEDDING_MODEL

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.NOTSET
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60
ALLOWED_EXTENSIONS: Set[str] = {
    '.txt', '.md', '.py', '.json', '.csv', '.html', 
    '.js', '.css', '.xml', '.yaml', '.yml', '.log', '.ini'
}

USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LAST_ACTIVITY: Dict[str, datetime] = {}
RATE_LIMIT_TOKENS: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX))

async def get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]

def check_rate_limit(user_id: str) -> bool:
    now = datetime.now()
    queue = RATE_LIMIT_TOKENS[user_id]
    
    while queue and queue[0] < now - timedelta(seconds=RATE_LIMIT_WINDOW):
        queue.popleft()
    
    if len(queue) < RATE_LIMIT_MAX:
        queue.append(now)
        return True
    return False

def update_activity(user_id: str):
    USER_LAST_ACTIVITY[user_id] = datetime.now()

async def read_file_content(file_path: str) -> str:
    try:
        def _read_safe():
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return None
                
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                
            try:
                return raw_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return raw_data.decode('latin-1')
                except UnicodeDecodeError:
                    return raw_data.decode('ascii', errors='ignore')

        content = await asyncio.to_thread(_read_safe)
        
        if content is None:
            return "[... File too large to process ...]"
            
        return content

    except Exception as e:
        logger.error(f"File read error: {e}")
        return "[Error reading file]"

async def send_chunked_response(update: Update, text: str):
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
            await update.message.reply_text(
                chunk, 
                parse_mode=ParseMode.MARKDOWN if '```' in chunk else None
            )
        except TelegramError:
            await update.message.reply_text(chunk, parse_mode=None)
            
        if text:
            await asyncio.sleep(0.5)

async def background_maintenance(context: ContextTypes.DEFAULT_TYPE):
    services = context.application.bot_data
    scheduler = services['scheduler']
    mem_mgr = services['mem_mgr']
    
    try:
        logger.info("Starting daily system maintenance...")
        
        deleted_count = await asyncio.to_thread(scheduler.cleanup_old_schedules, days_old=30)
        if deleted_count > 0:
            logger.info(f"Cleaned {deleted_count} old schedules.")
            
        await asyncio.to_thread(mem_mgr.optimize_memories, user_id=None)
        
        logger.info("Daily maintenance completed.")
            
    except Exception as e:
        logger.error(f"Maintenance task failed: {e}")

async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE):
    services = context.application.bot_data
    db = services['db']
    mem_mgr = services['mem_mgr']
    
    try:
        users = await asyncio.to_thread(db.get_cursor().execute, "SELECT DISTINCT user_id FROM memories WHERE status='active'")
        user_list = users.fetchall()
        
        for row in user_list:
            await asyncio.to_thread(mem_mgr.deduplicate_existing_memories, row[0])
            await asyncio.to_thread(mem_mgr.apply_decay_rules, row[0])
            
    except Exception as e:
        logger.error(f"Optimization task failed: {e}")

async def background_cleanup(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    timeout = timedelta(minutes=60)
    
    inactive_users = [
        uid for uid, last_time in USER_LAST_ACTIVITY.items()
        if now - last_time > timeout and uid in USER_LOCKS and not USER_LOCKS[uid].locked()
    ]
    
    for uid in inactive_users:
        USER_LOCKS.pop(uid, None)
        USER_LAST_ACTIVITY.pop(uid, None)
        RATE_LIMIT_TOKENS.pop(uid, None)

async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE):
    services = context.application.bot_data
    scheduler = services['scheduler']
    chat_handler = services['chat_handler']
    
    try:
        due_items = await asyncio.to_thread(scheduler.get_due_schedules, lookback_minutes=5)
        if not due_items:
            return

        for item in due_items:
            user_id = str(item['user_id'])
            schedule_id = item['id']
            ctx_text = item['context']
            
            try:
                ai_resp = await asyncio.to_thread(chat_handler.trigger_proactive_message, user_id, ctx_text)
                
                if ai_resp:
                    await context.bot.send_message(
                        chat_id=int(user_id),
                        text=ai_resp
                    )
                    await asyncio.to_thread(scheduler.mark_as_executed, schedule_id, "Sent")
                else:
                    logger.warning(f"Empty AI response for schedule {schedule_id}")
                    await asyncio.to_thread(scheduler.mark_as_executed, schedule_id, "Empty Response")
                    
            except Exception as e:
                logger.error(f"Schedule delivery failed for {schedule_id}: {e}")
                await asyncio.to_thread(scheduler.mark_as_executed, schedule_id, f"Failed: {e}")
                
    except Exception as e:
        logger.error(f"Scheduler checker loop failed: {e}")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    txt = (
        "👋 <b>Halo! Aku Vira.</b>\n\n"
        "Aku adalah kakak perempuan AI-mu. Aku bisa membantu mengingat hal-hal penting, "
        "mengatur jadwal, dan ngobrol santai.\n\n"
        "Gunakan /help untuk melihat daftar perintah."
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🤖 <b>DAFTAR PERINTAH</b>\n\n"
        "📝 <b>Core:</b>\n"
        "/reset - Mulai sesi chat baru\n"
        "/wipe - Hapus SEMUA data (Hard Reset)\n"
        "/forget [hal] - Lupakan memori spesifik\n\n"
        "🧠 <b>Memory & Schedule:</b>\n"
        "/memory [query] - Cari ingatan\n"
        "/schedule - Lihat jadwal aktif\n"
        "/summary - Lihat ringkasan jadwal (AI Generated)\n"
        "/remind [text] - Buat reminder cepat\n\n"
        "📦 <b>Data Management:</b>\n"
        "/export - Backup semua data\n"
        "/import - Restore dari backup\n"
        "/analytics - Lihat statistik penggunaan\n\n"
        "⚙️ <b>System:</b>\n"
        "/model [name] - Ganti model AI\n"
        "/stats - Status & Statistik"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    services = context.application.bot_data
    chat_handler = services['chat_handler']
    mem_mgr = services['mem_mgr']
    scheduler = services['scheduler']
    
    m_stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
    s_stats = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
    sys_stats = chat_handler.get_system_stats()
    
    msg = (
        f"🤖 <b>SYSTEM STATUS</b>\n\n"
        f"<b>Active Model:</b> <code>{chat_handler.chat_model_name}</code>\n"
        f"<b>API Health:</b> {sys_stats.get('api_health', 'Unknown')}\n\n"
        f"<b>User Data:</b>\n"
        f"• Memories: {m_stats.get('active', 0)} active\n"
        f"• Schedules: {s_stats.get('pending', 0)} pending\n"
        f"• Sessions: {sys_stats.get('sessions', 0)} cached"
    )
    
    kb = [[InlineKeyboardButton("📊 Full Stats", callback_data="refresh_stats")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    services = context.application.bot_data
    scheduler = services['scheduler']
    
    schedules = await asyncio.to_thread(scheduler.get_upcoming_schedules, user_id, limit=10)
    
    if not schedules:
        await update.message.reply_text("📅 Tidak ada jadwal aktif saat ini.")
        return

    msg = "🗓️ <b>JADWAL KAMU</b>\n\n" + "\n".join(schedules)
        
    kb = [
        [InlineKeyboardButton("📝 AI Summary", callback_data="get_schedule_summary")],
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_schedule")]
    ]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def cmd_schedule_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    services = context.application.bot_data
    session_manager = services['chat_handler'].session_manager
    
    summary = session_manager.get_schedule_summary(user_id)
    if not summary:
        summary = "Belum ada ringkasan jadwal yang dibuat."
        
    await update.message.reply_text(f"📋 <b>Ringkasan Jadwal (AI):</b>\n\n{summary}", parse_mode=ParseMode.HTML)

async def cmd_remind(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    if not context.args:
        await update.message.reply_text(
            "⚠️ Gunakan format: `/remind [waktu] [pesan]`\n"
            "Contoh: `/remind 30m beli susu` atau `/remind 2h meeting`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    services = context.application.bot_data
    scheduler = services['scheduler']
    chat_handler = services['chat_handler']
    
    try:
        text = " ".join(context.args)
        import re
        time_match = re.match(r'(\d+)([mh])\s+(.+)', text)
        
        if not time_match:
            await update.message.reply_text("❌ Format tidak valid. Contoh: `30m beli susu`", parse_mode=ParseMode.MARKDOWN)
            return
        
        amount = int(time_match.group(1))
        unit = time_match.group(2)
        message = time_match.group(3)
        
        now = datetime.now()
        if unit == 'm':
            trigger_time = now + timedelta(minutes=amount)
        else:
            trigger_time = now + timedelta(hours=amount)
        
        schedule_id = await asyncio.to_thread(
            scheduler.add_schedule,
            user_id=user_id,
            trigger_time=trigger_time,
            context=message,
            priority=1
        )
        
        if schedule_id:
            await asyncio.to_thread(chat_handler.executor.submit, chat_handler._refresh_schedule_summary, user_id)
            await update.message.reply_text(
                f"✅ Reminder diatur untuk <b>{trigger_time.strftime('%H:%M')}</b>\n"
                f"📝 {message}",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text("❌ Gagal membuat reminder.")
            
    except Exception as e:
        logger.error(f"Remind command error: {e}")
        await update.message.reply_text("❌ Terjadi kesalahan saat membuat reminder.")

async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    if not context.args:
        await update.message.reply_text(
            "⚠️ Gunakan format: `/forget [hal yang ingin dilupakan]`\n"
            "Contoh: `/forget novel`\n\n"
            "Gunakan /wipe untuk menghapus SEMUA data.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
        
    query = " ".join(context.args)
    services = context.application.bot_data
    mem_mgr = services['mem_mgr']
    analyzer = services['analyzer']
    
    status_msg = await update.message.reply_text("🔍 Mencari memori terkait...")
    
    try:
        emb = await asyncio.to_thread(analyzer.get_embedding, query)
        forgotten = await asyncio.to_thread(mem_mgr.forget_memory, user_id, query, emb)
        
        if forgotten:
            await status_msg.edit_text(f"✅ Berhasil melupakan: <b>'{forgotten}'</b>", parse_mode=ParseMode.HTML)
            await asyncio.to_thread(services['chat_handler'].clear_session, user_id)
        else:
            await status_msg.edit_text("❌ Tidak menemukan memori yang relevan atau spesifik.")
            
    except Exception as e:
        logger.error(f"Forget cmd error: {e}")
        await status_msg.edit_text("❌ Terjadi kesalahan sistem.")

async def cmd_wipe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    services = context.application.bot_data
    msg = await update.message.reply_text("💥 Menghapus semua data...")
    
    try:
        del_count = await asyncio.to_thread(services['mem_mgr'].wipe_all_memories, user_id)
        await asyncio.to_thread(services['chat_handler'].clear_session, user_id)
        
        services['db'].get_cursor().execute(
            "UPDATE schedules SET status='cancelled' WHERE user_id=? AND status='pending'", 
            (user_id,)
        )
        services['db'].commit()
        
        user_path = os.path.join("storage", "userdata", user_id)
        if os.path.exists(user_path):
            await asyncio.to_thread(shutil.rmtree, user_path, ignore_errors=True)
            
        await msg.edit_text(f"✅ Reset Selesai.\nDihapus: {del_count} memori & file user.")
    except Exception as e:
        logger.error(f"Wipe error: {e}")
        await msg.edit_text("❌ Reset gagal.")

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    status_msg = await update.message.reply_text("📦 Sedang mengumpulkan data...")
    
    services = context.application.bot_data
    db = services['db']
    
    try:
        def get_table_data(table_name):
            cursor = db.get_cursor()
            cursor.execute(f"SELECT * FROM {table_name} WHERE user_id=?", (user_id,))
            rows = cursor.fetchall()
            
            if not rows:
                return []
                
            cols = [description[0] for description in cursor.description]
            return [dict(zip(cols, row)) for row in rows]

        export_data = await asyncio.to_thread(lambda: {
            "meta": {
                "user_id": user_id,
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "memories": get_table_data("memories"),
            "schedules": get_table_data("schedules")
        })
        
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, bytes):
                return obj.hex()
            raise TypeError(f"Type {type(obj)} not serializable")

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False, encoding='utf-8') as tmp:
            json.dump(export_data, tmp, default=json_serializer, indent=2, ensure_ascii=False)
            tmp_path = tmp.name

        await update.message.reply_document(
            document=open(tmp_path, 'rb'),
            filename=f"vira_backup_{user_id}_{datetime.now().strftime('%Y%m%d')}.json",
            caption=(
                "📦 <b>BACKUP DATA SELESAI</b>\n\n"
                f"✅ Memories: {len(export_data['memories'])}\n"
                f"✅ Schedules: {len(export_data['schedules'])}\n\n"
                "<i>File ini berisi seluruh data mentah akunmu.</i>"
            ),
            parse_mode=ParseMode.HTML
        )
        
        await status_msg.delete()
        os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"Export failed for {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("❌ Gagal mengekspor data.")

async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    if not update.message.document:
        await update.message.reply_text(
            "⚠️ Kirim file backup JSON bersamaan dengan command `/import`\n"
            "Atau reply file backup dengan command ini.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    doc = update.message.document
    if not doc.file_name.endswith('.json'):
        await update.message.reply_text("❌ File harus berformat JSON.")
        return
    
    status_msg = await update.message.reply_text("📥 Memproses backup...")
    
    try:
        file_obj = await doc.get_file()
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
            await file_obj.download_to_drive(tmp.name)
            tmp_path = tmp.name
        
        with open(tmp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        os.remove(tmp_path)
        
        if not data.get('meta') or not data.get('memories') or not data.get('schedules'):
            await status_msg.edit_text("❌ Format backup tidak valid.")
            return
        
        services = context.application.bot_data
        db = services['db']
        cursor = db.get_cursor()
        
        imported_mem = 0
        imported_sched = 0
        
        for mem in data['memories']:
            try:
                embedding = bytes.fromhex(mem['embedding']) if mem.get('embedding') else None
                
                cursor.execute(
                    """INSERT OR REPLACE INTO memories 
                    (user_id, summary, memory_type, priority, embedding, status, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        mem.get('summary'),
                        mem.get('memory_type', 'general'),
                        mem.get('priority', 0.5),
                        embedding,
                        mem.get('status', 'active'),
                        mem.get('created_at'),
                        mem.get('last_used_at')
                    )
                )
                imported_mem += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")
        
        for sched in data['schedules']:
            try:
                cursor.execute(
                    """INSERT OR REPLACE INTO schedules 
                    (user_id, scheduled_at, context, priority, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        sched.get('scheduled_at'),
                        sched.get('context'),
                        sched.get('priority', 0),
                        sched.get('status', 'pending'),
                        sched.get('created_at')
                    )
                )
                imported_sched += 1
            except Exception as e:
                logger.warning(f"Failed to import schedule: {e}")
        
        db.commit()
        
        await status_msg.edit_text(
            f"✅ <b>IMPORT SELESAI</b>\n\n"
            f"📝 Memories: {imported_mem}\n"
            f"📅 Schedules: {imported_sched}",
            parse_mode=ParseMode.HTML
        )
        
    except json.JSONDecodeError:
        await status_msg.edit_text("❌ File JSON tidak valid.")
    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        await status_msg.edit_text("❌ Gagal mengimport data.")

async def cmd_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    services = context.application.bot_data
    db = services['db']
    
    status_msg = await update.message.reply_text("📊 Menganalisis data...")
    
    try:
        cursor = db.get_cursor()
        
        cursor.execute(
            "SELECT memory_type, COUNT(*) FROM memories WHERE user_id=? AND status='active' GROUP BY memory_type",
            (user_id,)
        )
        mem_by_type = dict(cursor.fetchall())
        
        cursor.execute(
            "SELECT MIN(created_at), MAX(last_used_at) FROM memories WHERE user_id=? AND status='active'",
            (user_id,)
        )
        mem_dates = cursor.fetchone()
        
        cursor.execute(
            "SELECT status, COUNT(*) FROM schedules WHERE user_id=? GROUP BY status",
            (user_id,)
        )
        sched_by_status = dict(cursor.fetchall())
        
        cursor.execute(
            "SELECT AVG(priority) FROM memories WHERE user_id=? AND status='active'",
            (user_id,)
        )
        row = cursor.fetchone()
        avg_priority = row[0] if row and row[0] is not None else 0
        
        # Sekarang values() adalah integer, jadi sum() aman dilakukan
        total_mem = sum(mem_by_type.values())
        total_sched = sum(sched_by_status.values())
        
        mem_type_str = "\n".join([f"  • {k}: {v}" for k, v in mem_by_type.items()]) if mem_by_type else "  Belum ada data"
        sched_status_str = "\n".join([f"  • {k}: {v}" for k, v in sched_by_status.items()]) if sched_by_status else "  Belum ada data"
        
        first_mem = mem_dates[0] if mem_dates and mem_dates[0] else "N/A"
        last_activity = mem_dates[1] if mem_dates and mem_dates[1] else "N/A"
        
        msg = (
            f"📊 <b>ANALYTICS DASHBOARD</b>\n\n"
            f"<b>📝 Memories Overview</b>\n"
            f"Total Active: {total_mem}\n"
            f"Average Priority: {avg_priority:.2f}\n"
            f"First Memory: {first_mem}\n"
            f"Last Activity: {last_activity}\n\n"
            f"<b>By Type:</b>\n{mem_type_str}\n\n"
            f"<b>📅 Schedules Overview</b>\n"
            f"Total: {total_sched}\n\n"
            f"<b>By Status:</b>\n{sched_status_str}"
        )
        
        await status_msg.edit_text(msg, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}", exc_info=True)
        await status_msg.edit_text("❌ Gagal menganalisis data.")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    services = context.application.bot_data
    await asyncio.to_thread(services['chat_handler'].clear_session, user_id)
    await update.message.reply_text("🧹 Sesi chat baru dimulai.")

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    services = context.application.bot_data
    
    query_text = " ".join(context.args) if context.args else None
    
    if query_text:
        results = await asyncio.to_thread(services['mem_mgr'].search_memories, user_id, query_text, limit=5)
        msg = f"🔎 <b>Search: {query_text}</b>\n\n"
    else:
        cursor = services['db'].get_cursor()
        cursor.execute(
            "SELECT summary, memory_type FROM memories WHERE user_id=? AND status='active' ORDER BY last_used_at DESC LIMIT 5",
            (user_id,)
        )
        results = [{"summary": r[0], "type": r[1]} for r in cursor.fetchall()]
        msg = "🧠 <b>Ingatan Terakhir</b>\n\n"

    if not results:
        msg += "Tidak ada data."
    else:
        for r in results:
            icon = "❤️" if r.get('type') == 'emotion' else "📌"
            msg += f"{icon} {r['summary'][:100]}\n"
            
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_memory")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    services = context.application.bot_data
    
    m = await asyncio.to_thread(services['mem_mgr'].get_memory_stats, user_id)
    s = await asyncio.to_thread(services['scheduler'].get_schedule_stats, user_id)
    d = await asyncio.to_thread(services['db'].get_database_stats)
    
    msg = (
        "📊 <b>STATISTIK DETAIL</b>\n\n"
        f"<b>User Memory:</b>\n"
        f"Active: {m.get('active', 0)}\n"
        f"Archived: {m.get('archived', 0)}\n\n"
        f"<b>Scheduler:</b>\n"
        f"Pending: {s.get('pending', 0)}\n"
        f"Executed: {s.get('executed', 0)}\n"
        f"Cancelled: {s.get('cancelled', 0)}\n\n"
        f"<b>System DB:</b>\n"
        f"Users: {d.get('total_users', 0)}\n"
        f"Size: {d.get('db_size_mb', 0)} MB"
    )
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    try:
        if data == "refresh_schedule":
            await cmd_schedule(update, context)
            await query.delete_message()
            
        elif data == "get_schedule_summary":
            await cmd_schedule_summary(update, context)
            
        elif data == "refresh_memory":
            await cmd_memory(update, context)
            await query.delete_message()
            
        elif data == "refresh_stats":
            await cmd_stats(update, context)
            await query.delete_message()
            
    except Exception as e:
        logger.error(f"Callback error: {e}")
        await query.edit_message_text("❌ Gagal memuat data.")

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id
    
    if not check_rate_limit(user_id):
        await update.message.reply_text("⏳ Tunggu sebentar, jangan spam ya.")
        return
        
    update_activity(user_id)
    
    lock = await get_user_lock(user_id)
    if lock.locked():
        await update.message.reply_text("⏳ Sebentar, Vira masih mikir...")
        return

    text_input = update.message.text or update.message.caption or ""
    img_path = None
    user_dir = os.path.join("storage", "userdata", user_id)
    
    if update.message.document:
        text_input = await handle_document(update, user_dir, text_input)
        if text_input is None:
            return

    if update.message.photo:
        img_path = await handle_photo(update, user_dir)
        if img_path is None:
            return

    if not text_input and not img_path:
        return

    async with lock:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        try:
            services = context.application.bot_data
            chat_handler = services['chat_handler']
            
            resp = await asyncio.to_thread(chat_handler.process_message, user_id, text_input, img_path)
            
            if not resp or resp == "ERROR":
                resp = "😵 Maaf, ada kesalahan sistem internal."
            
            await send_chunked_response(update, resp)
            
        except Exception as e:
            logger.error(f"Message process error: {e}", exc_info=True)
            await update.message.reply_text("😵 Terjadi error pada sistem.")
        
        finally:
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

async def handle_document(update: Update, user_dir: str, text_input: str) -> Optional[str]:
    doc = update.message.document
    fname = doc.file_name or "file.txt"
    ext = os.path.splitext(fname)[1].lower()
    
    if ext in ALLOWED_EXTENSIONS and doc.file_size <= MAX_FILE_SIZE:
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
        except Exception:
            await update.message.reply_text("❌ Gagal membaca file.")
            return None
    else:
        if not text_input:
            await update.message.reply_text("⚠️ File tidak didukung atau terlalu besar.")
            return None
    return text_input

async def handle_photo(update: Update, user_dir: str) -> Optional[str]:
    os.makedirs(user_dir, exist_ok=True)
    try:
        p_obj = await update.message.photo[-1].get_file()
        img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
        await p_obj.download_to_drive(img_path)
        return img_path
    except Exception:
        await update.message.reply_text("❌ Gagal mengunduh gambar.")
        return None

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update error: {context.error}")

async def post_init(app: Application):
    db = DBConnection()
    mem_mgr = MemoryManager(db)
    analyzer = MemoryAnalyzer()
    scheduler = SchedulerService(db)
    chat_handler = ChatHandler(mem_mgr, analyzer, scheduler)
    
    app.bot_data.update({
        'db': db,
        'mem_mgr': mem_mgr,
        'analyzer': analyzer,
        'scheduler': scheduler,
        'chat_handler': chat_handler
    })
    
    if app.job_queue:
        app.job_queue.run_repeating(background_maintenance, interval=86400, first=60)
        app.job_queue.run_repeating(background_schedule_checker, interval=60, first=10)
        app.job_queue.run_repeating(background_cleanup, interval=1800, first=300)
        app.job_queue.run_repeating(background_memory_optimization, interval=7200, first=3600)

    print(f"✅ System Ready. Model: {CHAT_MODEL}")

async def post_shutdown(app: Application):
    if 'db' in app.bot_data:
        app.bot_data['db'].close()
    print("🛑 System Shutdown.")

if __name__ == '__main__':
    if not TOKEN:
        exit("❌ TELEGRAM_BOT_TOKEN missing in .env")

    defaults = Defaults(parse_mode=ParseMode.HTML)

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .defaults(defaults)
        .concurrent_updates(True)
        .build()
    )
    
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(CommandHandler('reset', cmd_reset))
    app.add_handler(CommandHandler('wipe', cmd_wipe))
    app.add_handler(CommandHandler('memory', cmd_memory))
    app.add_handler(CommandHandler('forget', cmd_forget))
    app.add_handler(CommandHandler('schedule', cmd_schedule))
    app.add_handler(CommandHandler('summary', cmd_schedule_summary))
    app.add_handler(CommandHandler('remind', cmd_remind))
    app.add_handler(CommandHandler('export', cmd_export))
    app.add_handler(CommandHandler('import', cmd_import))
    app.add_handler(CommandHandler('analytics', cmd_analytics))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.Document.ALL) & (~filters.COMMAND),
        handle_msg
    ))
    
    app.add_error_handler(error_handler)
    
    print("🚀 Bot is running...")
    app.run_polling(drop_pending_updates=True)