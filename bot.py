"""Telegram long-polling entrypoint for the data-analyst agent bot."""
import json
import logging
import os
from collections import defaultdict, deque

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from agent import answer_question
from run_logger import RunLogger

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HISTORY_LEN = 10
chat_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
run_logger = RunLogger()


def _expects_final_answer(text: str) -> bool:
    """Every question (single-turn, or the last of a multi-turn sequence) spells out
    the exact reply shape it wants, which always includes a "log_url" key per the
    assignment spec. Earlier messages in a multi-turn thread are just context and
    don't ask for a reply yet."""
    return "log_url" in text.lower()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    text = update.message.text
    chat_history[chat_id].append(text)

    if not _expects_final_answer(text):
        # Context-building message in a multi-turn thread -- wait for the final one.
        return

    try:
        result = answer_question(list(chat_history[chat_id]), run_logger)
    except Exception:
        logger.exception("Agent failed to answer question")
        result = {"answer": None, "log_url": ""}

    # This exchange is complete -- don't let it leak into the next, unrelated question.
    chat_history[chat_id].clear()

    reply = json.dumps(result, ensure_ascii=False)
    await update.message.reply_text(reply)


def main() -> None:
    token = os.environ["BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot starting (long polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
