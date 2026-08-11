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
from tools.python_exec import reset_workspace, run_python
from tools.web_search import web_search

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
MAX_TURNS = 25
MAX_RETRIES = 6
DEFAULT_RETRY_DELAY = 5.0
MIN_CALL_INTERVAL = 10.0  # seconds; keeps us under free-tier RPM even without hitting 429
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
3d. Official statistics agencies publish their figures as large PDF reports on their own domain, \
so a good search names the issuing body, the report title, the period the question asks about, \
and filetype:pdf. Inside such a report the precise figures usually live in numbered detailed \
tables in an appendix, while the early chapters only summarise them in prose -- so locate the \
table that matches the question's exact wording (period, age group, rural/urban, etc.) and read \
the value from it, rather than quoting a rounded number from the narrative.
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
call do NOT exist in the next one, so every call must import everything it uses. FILES, however, \
DO persist: all calls for this question share one working directory. Use that for anything \
expensive: download a PDF/CSV once with `open("source.pdf","wb").write(r.content)`, then in later \
calls open that saved file directly instead of re-downloading it. Re-downloading the same large \
document every call is the single biggest waste of time -- save it once, then parse it.
8. Prefer few, substantial calls over many tiny ones. When you open a large PDF, extract and \
print everything you might need in that same call (e.g. scan pages and print every line matching \
the statistic you want, with page numbers) rather than making one call per page or per guess.
9. Big PDFs: scanning every page of a 300-page report will hit the time limit. Work in slices -- \
download once, then parse a range of pages per call (e.g. pages 0-40, then 40-80), printing \
matches as you go. If a call times out, the file is still on disk and any output printed before \
the timeout is returned to you: continue from where you left off rather than restarting. If a \
government site fails with an SSL/certificate error, retry that request with verify=False.
10. Never decide the answer first and then search to confirm it. If you catch yourself searching \
for a candidate answer you already have in mind, stop -- that only finds agreement, not truth. \
Read the value out of the primary document instead. A news article quoting a figure is not the \
source document; fall back to it only after the primary source has genuinely failed.
11. Everything above is generic method, not a description of the task. The only statement of what \
you must answer is the user's message itself: re-read it before your first tool call and before \
your final answer, and make sure what you are actually looking up is what it asked for. Any topic, \
agency, or example phrasing mentioned in these instructions is illustrative and must never be \
treated as the subject of the question.
"""

RUN_PYTHON_DECL = types.FunctionDeclaration(
    name="run_python",
    description=(
        "Execute Python code in a sandboxed subprocess. requests, json, io, re, urllib.parse "
        "are preimported (plus a `headers` dict with a default User-Agent). pandas, numpy, "
        "BeautifulSoup (bs4), and pdfplumber are installed but NOT preimported -- `import` "
        "them yourself only in calls that actually need them, to keep memory use down. Use "
        "print() to see output. "
        "IMPORTANT: each call is a new process, so variables and imports do NOT carry over -- "
        "import everything the call uses. FILES DO carry over: every call for this question runs "
        "in the same working directory, so save a downloaded document to disk once (e.g. "
        "open('source.pdf','wb').write(r.content)) and open that file in later calls instead of "
        "re-downloading it."
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
    # Files written by this question's run_python calls are shared between those calls but
    # must not survive into the next question's run -- see tools.python_exec.reset_workspace.
    reset_workspace()
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
            if isinstance(final_answer, dict) and "answer" in final_answer and set(final_answer) <= {"answer", "log_url"}:
                # Model wrapped its output in the outer {"answer": ..., "log_url": ...}
                # envelope (with or without log_url) instead of just the "answer" value --
                # unwrap it.
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
