from telegram.ext import ContextTypes


def get_brain_from_context(context: ContextTypes.DEFAULT_TYPE):
    return context.bot_data.get("brain")


