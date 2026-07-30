"""Fetches a public URL's raw content for the agent (datasets, HTML pages, etc.)."""
import requests

TIMEOUT_SECONDS = 20
MAX_BYTES = 2_000_000  # 2MB cap
USER_AGENT = "data-analyst-telegram-bot/1.0"


def fetch_url(url: str) -> dict:
    try:
        resp = requests.get(
            url,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        resp.raise_for_status()
        content = resp.raw.read(MAX_BYTES + 1, decode_content=True)
        truncated = len(content) > MAX_BYTES
        content = content[:MAX_BYTES]
        try:
            text = content.decode(resp.encoding or "utf-8", errors="replace")
        except (LookupError, TypeError):
            text = content.decode("utf-8", errors="replace")
        return {
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "text": text,
            "truncated": truncated,
            "error": None,
        }
    except requests.RequestException as exc:
        return {"status_code": None, "content_type": "", "text": "", "truncated": False, "error": str(exc)}
