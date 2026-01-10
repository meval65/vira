import os
import logging
import asyncio
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, Deque, Optional, Set
from collections import defaultdict, deque
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
from src.config import get_available_chat_models, CHAT_MODEL, EMBEDDING_MODEL

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

db = None
mem_mgr = None
chat_handler = None
scheduler = None

USER_LOCKS: Dict[str, asyncio.Lock] = {}
USER_LAST_ACTIVITY: Dict[str, datetime] = {}
RATE_LIMIT_TOKENS: Dict[str, Deque[datetime]] = defaultdict(deque)

MAX_FILE_SIZE = 5 * 1024 * 1024
RATE_LIMIT_MAX = 15
RATE_LIMIT_WINDOW = 4
ALLOWED_EXTENSIONS: Set[str] = {
    '.txt', '.md', '.py', '.json', '.csv', '.html', 
    '.js', '.css', '.xml', '.yaml', '.yml', '.log'
}

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
        with open(file_path, 'rb') as f:
            content = f.read(MAX_FILE_SIZE + 1024)
            
        if len(content) > MAX_FILE_SIZE:
            return "[... File too large, truncated ...]"
            
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('latin-1')
    except Exception:
        return ""

async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE):
    try:
        users = await asyncio.to_thread(db.get_cursor().execute, "SELECT DISTINCT user_id FROM memories WHERE status='active'")
        user_list = users.fetchall()
        
        for row in user_list:
            await asyncio.to_thread(mem_mgr.deduplicate_existing_memories, row[0])
            
    except Exception as e:
        logger.error(f"Optimization task failed: {e}")

async def background_cleanup(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now()
    timeout = timedelta(minutes=30)
    inactive_users = [
        uid for uid, last_time in USER_LAST_ACTIVITY.items()
        if now - last_time > timeout and uid in USER_LOCKS and not USER_LOCKS[uid].locked()
    ]
    
    for uid in inactive_users:
        USER_LOCKS.pop(uid, None)
        USER_LAST_ACTIVITY.pop(uid, None)
        RATE_LIMIT_TOKENS.pop(uid, None)

async def background_maintenance(context: ContextTypes.DEFAULT_TYPE):
    try:
        await asyncio.to_thread(mem_mgr.apply_decay_rules)
        await asyncio.to_thread(mem_mgr.optimize_memories, user_id=None)
        await asyncio.to_thread(scheduler.cleanup_old_schedules, days_old=30)
    except Exception as e:
        logger.error(f"Maintenance task failed: {e}")

async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE):
    try:
        due_items = await asyncio.to_thread(scheduler.get_due_schedules, lookback_minutes=5)
        if not due_items:
            return

        for item in due_items:
            user_id = item['user_id']
            schedule_id = item['id']
            ctx_text = item['context']
            
            try:
                ai_resp = await asyncio.to_thread(chat_handler.trigger_proactive_message, user_id, ctx_text)
                if ai_resp:
                    await context.bot.send_message(
                        chat_id=int(user_id),
                        text=ai_resp,
                        parse_mode=ParseMode.HTML if '<' in ai_resp else None
                    )
                    await asyncio.to_thread(scheduler.mark_as_executed, schedule_id, "Sent")
            except Exception as e:
                logger.error(f"Schedule delivery failed for {schedule_id}: {e}")
                await asyncio.to_thread(scheduler.mark_as_executed, schedule_id, f"Failed: {e}")
                
    except Exception as e:
        logger.error(f"Scheduler checker failed: {e}")

async def send_chunked_response(update: Update, text: str):
    if not text: 
        return
    
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML if '<' in chunk else None)
        if len(chunks) > 1:
            await asyncio.sleep(0.3)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    m_stats = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
    s_stats = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
    sys_stats = chat_handler.get_system_stats()
    
    msg = (
        f"🤖 <b>SYSTEM STATUS</b>\n\n"
        f"<b>Models:</b>\n"
        f"Chat: <code>{chat_handler.chat_model_name}</code>\n"
        f"Embed: <code>{EMBEDDING_MODEL}</code>\n\n"
        f"<b>Memory:</b> {m_stats.get('active', 0)} active | {m_stats.get('archived', 0)} archived\n"
        f"<b>Schedule:</b> {s_stats.get('pending', 0)} pending\n"
        f"<b>Sessions:</b> {sys_stats.get('active_sessions', 0)} active"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    history = await asyncio.to_thread(scheduler.get_schedule_history, user_id, limit=10, include_pending=True)
    
    if not history:
        await update.message.reply_text("📅 No active schedules.")
        return

    msg = "🗓️ <b>SCHEDULES</b>\n\n"
    for item in history:
        status_icon = "⏳" if item['status'] == 'pending' else "✅"
        ts = item['scheduled_at']
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts).strftime("%d/%m %H:%M")
            except ValueError: pass
        msg += f"{status_icon} <b>{ts}</b>: {item['context'][:50]}\n"
        
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_schedule")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    data = query.data
    
    try:
        if data == "refresh_schedule":
            await cmd_schedule(update, context)
            await query.delete_message()
            
        elif data == "refresh_memory":
            await cmd_memory(update, context)
            await query.delete_message()
            
        elif data == "refresh_stats":
            await cmd_stats(update, context)
            await query.delete_message()
            
    except Exception:
        await query.edit_message_text("❌ Data refresh failed.")

async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    msg = await update.message.reply_text("💥 Wiping data...")
    
    try:
        del_count = await asyncio.to_thread(mem_mgr.wipe_all_memories, user_id)
        await asyncio.to_thread(chat_handler.clear_session, user_id)
        
        db.get_cursor().execute(
            "UPDATE schedules SET status='cancelled' WHERE user_id=? AND status='pending'", 
            (user_id,)
        )
        db.commit()
        
        user_path = os.path.join("storage", "userdata", user_id)
        if os.path.exists(user_path):
            await asyncio.to_thread(shutil.rmtree, user_path, ignore_errors=True)
            
        await msg.edit_text(f"✅ Reset complete.\nDeleted {del_count} memories.")
    except Exception:
        await msg.edit_text("❌ Reset failed.")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    await asyncio.to_thread(chat_handler.clear_session, user_id)
    await update.message.reply_text("🧹 Context cleared.")

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    query_text = context.args[0] if context.args else None
    
    if query_text:
        results = await asyncio.to_thread(mem_mgr.search_memories, user_id, query_text, limit=8)
        msg = f"🔎 <b>Search: {query_text}</b>\n\n"
    else:
        cursor = db.get_cursor()
        cursor.execute(
            "SELECT summary, memory_type FROM memories WHERE user_id=? AND status='active' ORDER BY last_used_at DESC LIMIT 8",
            (user_id,)
        )
        results = [{"summary": r[0], "type": r[1]} for r in cursor.fetchall()]
        msg = "🧠 <b>Recent Memories</b>\n\n"

    if not results:
        msg += "No data found."
    else:
        for r in results:
            msg += f"• [{r['type'].upper()}] {r['summary'][:80]}...\n"
            
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_memory")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    models = get_available_chat_models()
    if not context.args:
        curr = chat_handler.chat_model_name
        list_str = "\n".join([f"- {m}" for m in models])
        await update.message.reply_text(f"Current: <code>{curr}</code>\nAvailable:\n{list_str}", parse_mode=ParseMode.HTML)
        return

    target = context.args[0]
    if target in models:
        await asyncio.to_thread(chat_handler.change_model, target)
        await update.message.reply_text(f"✅ Model changed to <code>{target}</code>", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("❌ Invalid model name.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "🤖 <b>VIRA COMMANDS</b>\n\n"
        "/start - Init bot\n"
        "/reset - New chat session\n"
        "/memory [query] - Search memory\n"
        "/schedule - Check reminders\n"
        "/model [name] - Switch AI\n"
        "/stats - View statistics\n"
        "/forget - Hard reset"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    update_activity(user_id)
    
    m = await asyncio.to_thread(mem_mgr.get_memory_stats, user_id)
    s = await asyncio.to_thread(scheduler.get_schedule_stats, user_id)
    d = await asyncio.to_thread(db.get_database_stats)
    
    msg = (
        "📊 <b>STATISTICS</b>\n\n"
        f"<b>User Memory:</b>\n"
        f"Active: {m.get('active', 0)}\n"
        f"Archived: {m.get('archived', 0)}\n"
        f"Avg Priority: {m.get('avg_priority', 0)}\n\n"
        f"<b>Scheduler:</b>\n"
        f"Pending: {s.get('pending', 0)}\n"
        f"Total Executed: {s.get('executed', 0)}\n\n"
        f"<b>Global DB:</b>\n"
        f"Users: {d.get('total_users', 0)}\n"
        f"Size: {d.get('db_size_mb', 0)} MB"
    )
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")]]
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))

async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id
    
    if not check_rate_limit(user_id):
        await update.message.reply_text("⏳ Rate limit exceeded.")
        return
        
    update_activity(user_id)
    
    lock = await get_user_lock(user_id)
    if lock.locked():
        await update.message.reply_text("⏳ Processing previous request...")
        return

    text_input = update.message.text or update.message.caption or ""
    img_path = None
    user_dir = os.path.join("storage", "userdata", user_id)
    os.makedirs(user_dir, exist_ok=True)

    if update.message.document:
        doc = update.message.document
        fname = doc.file_name or "file.txt"
        ext = os.path.splitext(fname)[1].lower()
        
        if ext in ALLOWED_EXTENSIONS and doc.file_size <= MAX_FILE_SIZE:
            try:
                f_obj = await doc.get_file()
                tmp = os.path.join(user_dir, f"tmp_{uuid.uuid4().hex[:8]}{ext}")
                await f_obj.download_to_drive(tmp)
                content = await read_file_content(tmp)
                text_input += f"\n\n[FILE: {fname}]\n{content}\n[END FILE]"
                if os.path.exists(tmp): os.remove(tmp)
            except Exception:
                await update.message.reply_text("❌ File processing failed.")
                return
        else:
            if not text_input:
                await update.message.reply_text("⚠️ Unsupported file or size limit.")
                return

    if update.message.photo:
        try:
            p_obj = await update.message.photo[-1].get_file()
            img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
            await p_obj.download_to_drive(img_path)
        except Exception:
            await update.message.reply_text("❌ Image download failed.")
            return

    if not text_input and not img_path:
        return

    async with lock:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        try:
            resp = await asyncio.to_thread(chat_handler.process_message, user_id, text_input, img_path)
            
            if not resp or resp == "ERROR":
                resp = "😵 Internal processing error."
            
            await send_chunked_response(update, resp)
            
        except Exception as e:
            logger.error(f"Message process error: {e}")
            await update.message.reply_text("😵 System error occurred.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update error: {context.error}")

async def post_init(app: Application):
    global db, mem_mgr, chat_handler, scheduler
    
    db = DBConnection()
    mem_mgr = MemoryManager(db)
    analyzer = MemoryAnalyzer()
    scheduler = SchedulerService(db)
    chat_handler = ChatHandler(mem_mgr, analyzer, scheduler)
    
    if app.job_queue:
        app.job_queue.run_repeating(background_maintenance, interval=86400, first=300)
        app.job_queue.run_repeating(background_schedule_checker, interval=120, first=15)
        app.job_queue.run_repeating(background_cleanup, interval=1800, first=600)
        app.job_queue.run_repeating(background_memory_optimization, interval=7200, first=3600)

    print(f"System Ready. Model: {CHAT_MODEL}")

async def post_shutdown(app: Application):
    if db: db.close()
    print("System Shutdown.")

if __name__ == '__main__':
    if not TOKEN:
        exit("TELEGRAM_BOT_TOKEN missing.")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .concurrent_updates(True)
        .build()
    )
    
    app.add_handler(CommandHandler('start', cmd_help))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(CommandHandler('reset', cmd_reset))
    app.add_handler(CommandHandler('newchat', cmd_reset))
    app.add_handler(CommandHandler('memory', cmd_memory))
    app.add_handler(CommandHandler('model', cmd_model))
    app.add_handler(CommandHandler('forget', cmd_forget))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('stats', cmd_stats))
    app.add_handler(CommandHandler('jadwal', cmd_schedule))
    app.add_handler(CommandHandler('schedule', cmd_schedule))
    
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.Document.ALL) & (~filters.COMMAND),
        handle_msg
    ))
    
    app.add_error_handler(error_handler)
    
    app.run_polling(drop_pending_updates=True)