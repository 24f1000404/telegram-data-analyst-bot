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

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
MAX_TURNS = 20
MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 5.0

SYSTEM_PROMPT = """You are a data-analyst agent. You receive a Telegram message containing \
one data-analysis question. The message itself spells out the exact JSON shape the final \
answer must take (e.g. {"answer": {"state": "<state name>"}, "log_url": "..."}).

Rules:
1. Read the message carefully and identify the exact JSON shape requested for the "answer" \
value. This shape varies per question -- never assume a fixed format.
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
4. When you have the final answer, respond with ONLY a JSON object of the exact requested \
shape for the "answer" key -- do not include "log_url" yourself (the caller adds it), do not \
wrap it in markdown code fences, and do not add any explanation text. Just the raw JSON object \
that belongs in "answer", e.g. {"state": "Assam"}.
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

TOOLS = [types.Tool(function_declarations=[RUN_PYTHON_DECL, FETCH_URL_DECL])]

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
    for attempt in range(MAX_RETRIES):
        try:
            return chat.send_message(content)
        except genai_errors.ClientError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(_extract_retry_delay(exc) + 1)


def _run_tool(name: str, tool_input: dict) -> dict:
    if name == "run_python":
        return run_python(tool_input["code"])
    if name == "fetch_url":
        return fetch_url(tool_input["url"])
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
                final_answer = {"raw": raw_text}
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
            trace["tool_calls"].append({"tool": fc.name, "input": tool_input, "result": result})
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
