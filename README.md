# Data-Analyst Telegram Bot

A Telegram bot backed by a Gemini tool-use agent. It answers ad-hoc data-analysis
questions (including ones pointing at public datasets like MOSPI) and replies with
exactly one JSON object: `{"answer": ..., "log_url": "..."}`.

## How it works

- `bot.py` — long-polls Telegram for messages, keeps a short per-chat history for
  multi-turn questions, and always answers the latest message.
- `agent.py` — a Gemini (`gemini-flash-lite-latest`, free tier) tool-use loop. It reads the exact JSON
  shape requested in the message, uses tools to compute a real answer, and returns
  only that JSON shape.
- `tools/python_exec.py` — runs agent-written Python in a sandboxed subprocess
  (temp dir, 30s timeout, `pandas`/`numpy`/`requests` preimported) so the agent can
  fetch and analyze real data instead of guessing.
- `tools/fetch.py` — fetches a URL's raw text/CSV/JSON/HTML content directly.
- `run_logger.py` — appends every run as one JSON line to `run.jsonl` and publishes
  the file via a public GitHub Gist, giving a stable, wget-able `log_url`.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in:
   - `BOT_TOKEN` — from [@BotFather](https://t.me/BotFather) (`/newbot`, username must end in `bot`)
   - `GEMINI_API_KEY` — free-tier key from [Google AI Studio](https://aistudio.google.com/apikey)
   - `GITHUB_TOKEN` — a GitHub personal access token with `gist` scope
   - `GITHUB_USERNAME` — your GitHub username
   - `GIST_ID` — leave blank on first run; the bot creates a gist and logs its id,
     then set `GIST_ID` to that value so the log URL stays stable across restarts
4. `python bot.py`

## Testing locally

Message your bot on Telegram with a data-analysis question and confirm the reply
is exactly one JSON object, e.g.:

```
{"answer": {"state": "Assam"}, "log_url": "https://gist.githubusercontent.com/.../raw/run.jsonl"}
```

Verify the log is fetchable:

```
wget -qO- <log_url>
```

To run the official grading harness against the bot locally, clone
[tds-p1-t2-2026-telegram-bot](https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot),
add your own questions to `evals/questions.json`, and point it at this bot's
username per that repo's instructions.

## Deployment (Fly.io)

Fly.io is recommended because this bot only makes outbound long-polling requests
to Telegram — it needs an always-on process, not an inbound HTTP endpoint — and
Fly's free allowance doesn't idle-sleep the way some PaaS free web services do.

```
fly launch --no-deploy   # creates the app from fly.toml, skip auto-deploy
fly secrets set BOT_TOKEN=... GEMINI_API_KEY=... GITHUB_TOKEN=... GITHUB_USERNAME=... GIST_ID=...
fly deploy
fly logs   # confirm "Bot starting (long polling)..." and no crash loop
```

Keep the app running through grading; redeploy (`fly deploy`) after any code change.
