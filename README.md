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

## Deployment (Render, no card required)

Render's free web-service tier doesn't require a credit card. Free services do
idle-sleep after 15 minutes without inbound HTTP traffic, so `bot.py` runs a tiny
health-check HTTP server (`GET /` -> `200 ok`) alongside the Telegram long-polling
loop — pair it with a free uptime pinger (e.g. [UptimeRobot](https://uptimerobot.com)
or [cron-job.org](https://cron-job.org)) hitting your Render URL every ~10 minutes
to keep the instance awake.

Steps:
1. On [render.com](https://render.com), **New** -> **Web Service**, connect this
   GitHub repo. Render auto-detects `render.yaml` / the `Dockerfile`.
2. In the service's **Environment** tab, set `BOT_TOKEN`, `GEMINI_API_KEY`,
   `GITHUB_TOKEN`, `GITHUB_USERNAME`, `GIST_ID` (same values as your local `.env`).
3. Deploy. Check the logs for `"Bot starting (long polling)..."` and no crash loop.
4. Add your Render service's public URL to an uptime pinger so it never sleeps.

(A `fly.toml` is also included if you'd rather use Fly.io — it doesn't idle-sleep,
but Fly now requires a card on file for account verification.)
```

Keep the app running through grading; redeploy (`fly deploy`) after any code change.
