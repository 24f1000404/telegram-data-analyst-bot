"""Telegram long-polling entrypoint for the data-analyst agent bot."""
import json
import logging
import os
import threading
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

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


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # keep noisy health-check hits out of the bot log


def _start_health_server():
    """Minimal HTTP endpoint so PaaS free tiers (e.g. Render) treat this as a live
    web service and an external uptime pinger can keep it from sleeping. The bot
    itself only does outbound long-polling and needs no inbound HTTP otherwise."""
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health check server listening on :%d", port)


def main() -> None:
    _start_health_server()
    token = os.environ["BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot starting (long polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
