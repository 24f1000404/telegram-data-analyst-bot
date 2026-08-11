"""Gemini tool-use agent that answers a data-analysis question with a single JSON object."""
import json
import os
import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from run_logger import RunLogger
from tools.fetch import fetch_url
from tools.python_exec import run_python
from tools.web_search import web_search

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
MAX_TURNS = 25
MAX_RETRIES = 6
DEFAULT_RETRY_DELAY = 5.0
MIN_CALL_INTERVAL = 4.5  # seconds; keeps us under free-tier RPM even without hitting 429
_last_call_ts = 0.0
LOG_STRING_LIMIT = 1000  # keeps run.jsonl (a GitHub Gist, ~1MB raw-view cap) from bloating


def _truncate_for_log(obj):
    """Recursively truncates long strings before writing a tool call into the run log --
    the model already sees the full (differently-capped) result; the log only needs enough
    to debug what happened."""
    if isinstance(obj, str):
        if len(obj) > LOG_STRING_LIMIT:
            return obj[:LOG_STRING_LIMIT] + f"... [truncated, {len(obj)} chars total]"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_for_log(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_for_log(v) for v in obj]
    return obj

SYSTEM_PROMPT = """You are a data-analyst agent. You receive a Telegram message containing \
one data-analysis question. The message itself spells out the exact JSON shape the final \
answer must take (e.g. {"answer": {"state": "<state name>"}, "log_url": "..."}).

Rules:
1. Read the message carefully and identify the exact JSON shape requested for the "answer" \
value. This shape varies per question and is NOT always an object -- it can be a plain string \
(e.g. `"answer": "..."` means reply with a bare JSON string like "2.8%", not an object wrapping \
it), a number, a list, or an object with specific keys (e.g. `"answer": {"state": "..."}`). \
Match the literal placeholder shown in the message exactly -- never wrap a requested plain \
string/number in an object, and never flatten a requested object into a plain value.
2. Use the run_python tool to actually compute the answer: fetch data with `requests`, and \
`import pandas as pd` when you need it to parse/analyze tabular data. For HTML pages, \
`from bs4 import BeautifulSoup` instead of hand-written regex/string parsing. For PDF reports \
(e.g. RBI/government annual reports), download the PDF bytes with `requests` and \
`import pdfplumber` to parse them -- most official statistics are published as PDF or HTML, \
not clean CSVs, so expect to need these. pandas, numpy, BeautifulSoup, and pdfplumber are all \
installed and importable, but NOT preimported -- import only the ones a given call actually \
needs (they're memory-heavy; importing all of them on every call wastes memory for no reason \
on lightweight calls like a simple web search). Use the fetch_url tool for a quick look at a \
page/CSV/JSON before writing code, if useful.
3. Do not fabricate numbers or facts. If the message points at a public dataset (e.g. MOSPI), \
locate and load the real data via code before answering.
3a. Use the web_search tool to find the primary source, never hand-write code to scrape \
google.com/search, bing.com, or duckduckgo.com -- those pages are JS-rendered or bot-walled and \
requests/BeautifulSoup cannot reliably read them; writing that code just wastes turns. Once \
web_search gives you the primary document's URL (the issuing organization's own site, e.g. \
rbi.org.in, mospi.gov.in), use fetch_url or run_python to actually fetch and read that source \
yourself -- never answer from a search snippet alone.
3b. If web_search result snippets disagree on a number, that is a signal to open the primary \
document and read the exact figure yourself -- not a reason to keep searching for a tie-breaker. \
Never make more than 2-3 web_search calls in a row before opening a source document.
3c. If a government site's navigation page only exposes JS-postback links (e.g. \
javascript:__doPostBack(...), common on rbi.org.in) instead of plain hrefs, that page's links \
are not reachable via requests -- go back to web_search with a more specific query (e.g. \
including the report title, "filetype:pdf", or a chapter/table name) to find a direct content \
URL instead of trying to parse the postback page.
4. When you have the final answer, respond with ONLY the JSON value of the exact requested \
shape for the "answer" key -- do not include "log_url" yourself (the caller adds it), do not \
wrap it in markdown code fences, and do not add any explanation text. Just the raw JSON value \
that belongs in "answer": that may be an object like {"state": "Assam"}, or it may be a bare \
value like "2.8%" or 42 if that's what the message's placeholder shows -- do not add object \
wrapping that the message didn't ask for.
4a. Your response must be valid, parseable JSON on its own, because it is parsed with a JSON \
parser, not read as English. If the answer is a string, it MUST be wrapped in double quotes -- \
respond with "2.8%" (7 characters including the quotes), never the bare text 2.8% (without \
quotes), which is not valid JSON and will be treated as a parse failure. If the answer is a \
number, respond with a bare unquoted number like 42 or 2.8. If unsure whether the requested \
shape is a quoted string or a bare number, match the placeholder's own quoting in the message \
exactly: "..." in the placeholder means quoted string, a bare word like <number> means unquoted.
5. If the question is part of a multi-turn conversation, answer only the latest message, using \
earlier turns as context only if the latest message clearly depends on them (e.g. it says \
"that dataset", "the same state", "it", etc). If the latest message is fully self-contained \
(states its own complete question and its own JSON shape), ignore earlier turns entirely -- \
they are unrelated leftover context, not part of this question.
6. Never reuse a numeric value, name, or answer from an earlier turn unless you recomputed it \
for the current question -- always derive the answer fresh from real data for the current message.
7. Each run_python call is a fresh, independent process -- variables and imports from a previous \
call do NOT exist in the next one. Write each call as a complete, self-contained script: import \
everything it uses, and re-fetch/recompute any data it needs, even if you already did so in an \
earlier call.
"""

RUN_PYTHON_DECL = types.FunctionDeclaration(
    name="run_python",
    description=(
        "Execute Python code in a sandboxed subprocess. requests, json, io, re, urllib.parse "
        "are preimported (plus a `headers` dict with a default User-Agent). pandas, numpy, "
        "BeautifulSoup (bs4), and pdfplumber are installed but NOT preimported -- `import` "
        "them yourself only in calls that actually need them, to keep memory use down. Use "
        "print() to see output. "
        "IMPORTANT: each call is a BRAND NEW process -- nothing persists between calls. "
        "Variables, imports, and downloaded data from a previous call are NOT available in the "
        "next one. Every call must be fully self-contained: import everything it uses and "
        "re-fetch or recompute any data it needs, even if an earlier call already did so."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"code": types.Schema(type="STRING", description="Python source to execute.")},
        required=["code"],
    ),
)

FETCH_URL_DECL = types.FunctionDeclaration(
    name="fetch_url",
    description=(
        "Fetch a public URL's raw text content (HTML/CSV/JSON). Not suitable for binary "
        "formats like .xlsx -- use run_python with pandas/requests instead."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"url": types.Schema(type="STRING")},
        required=["url"],
    ),
)

WEB_SEARCH_DECL = types.FunctionDeclaration(
    name="web_search",
    description=(
        "Search the web via a real search API (not scraping) and get back title/url/snippet "
        "results. Use this to locate the primary source (the issuing organization's own page or "
        "document) for a question -- then fetch that URL directly with fetch_url or run_python. "
        "Do not treat snippets themselves as the final answer."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="Search query.")},
        required=["query"],
    ),
)

TOOLS = [types.Tool(function_declarations=[RUN_PYTHON_DECL, FETCH_URL_DECL, WEB_SEARCH_DECL])]

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _extract_retry_delay(exc: genai_errors.ClientError) -> float:
    try:
        details = exc.details.get("error", {}).get("details", [])
    except AttributeError:
        details = []
    for d in details:
        if d.get("@type", "").endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s?", d.get("retryDelay", ""))
            if match:
                return float(match.group(1))
    return DEFAULT_RETRY_DELAY


def _send_message_with_retry(chat, content):
    global _last_call_ts
    for attempt in range(MAX_RETRIES):
        wait = MIN_CALL_INTERVAL - (time.time() - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        try:
            response = chat.send_message(content)
            _last_call_ts = time.time()
            return response
        except genai_errors.ClientError as exc:
            _last_call_ts = time.time()
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_extract_retry_delay(exc) + 1)


def _run_tool(name: str, tool_input: dict) -> dict:
    if name == "run_python":
        return run_python(tool_input["code"])
    if name == "fetch_url":
        return fetch_url(tool_input["url"])
    if name == "web_search":
        return web_search(tool_input["query"])
    return {"error": f"Unknown tool {name}"}


def answer_question(message_history: list[str], logger: RunLogger) -> dict:
    """Runs the agent loop for the latest message (message_history[-1]).

    Returns {"answer": <parsed answer value>, "log_url": <str>}.
    """
    client = _get_client()
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=TOOLS),
    )

    convo_text = "\n---\n".join(message_history)
    trace = {"messages_in": message_history, "tool_calls": []}

    final_answer = None
    response = _send_message_with_retry(chat, convo_text)
    for _ in range(MAX_TURNS):
        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            raw_text = "".join(p.text for p in parts if p.text).strip()
            raw_text = _strip_code_fence(raw_text)
            try:
                final_answer = json.loads(raw_text)
            except json.JSONDecodeError:
                # Model emitted a bare, unquoted string instead of valid JSON (e.g. `2.7%`
                # instead of `"2.7%"`) -- that's still unambiguously a plain-string answer,
                # so use it as-is rather than wrapping it in an object the question never asked
                # for.
                final_answer = raw_text
            if isinstance(final_answer, dict) and "log_url" in final_answer and "answer" in final_answer:
                # Model echoed the full {"answer": ..., "log_url": ...} wrapper instead
                # of just the "answer" value -- unwrap it.
                final_answer = final_answer["answer"]
            trace["final_text"] = raw_text
            break

        response_parts = []
        for fc in function_calls:
            tool_input = dict(fc.args)
            result = _run_tool(fc.name, tool_input)
            trace["tool_calls"].append({
                "tool": fc.name,
                "input": _truncate_for_log(tool_input),
                "result": _truncate_for_log(result),
            })
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.dumps(result)[:8000]},
                )
            )
        response = _send_message_with_retry(chat, response_parts)
    else:
        final_answer = {"error": "max_turns_exceeded"}
        trace["final_text"] = None

    log_url = logger.log_run({**trace, "final_answer": final_answer})
    return {"answer": final_answer, "log_url": log_url}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
