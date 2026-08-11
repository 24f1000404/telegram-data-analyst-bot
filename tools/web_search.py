"""Web search tool backed by the Tavily search API (not scraped search-engine
HTML, which is unreliable -- Google/Bing render results client-side and
DuckDuckGo bot-walls plain requests).
"""
import os

import requests

TIMEOUT_SECONDS = 15
MAX_RESULTS = 5


def web_search(query: str) -> dict:
    """Returns {"results": [{"title", "url", "snippet"}], "error": str|None}."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"results": [], "error": "TAVILY_API_KEY not configured"}
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": MAX_RESULTS},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]
        return {"results": results, "error": None}
    except requests.RequestException as exc:
        return {"results": [], "error": str(exc)}
