"""Sandboxed Python execution tool for the agent.

Runs agent-authored code in a fresh subprocess, with a wall-clock timeout and
captured stdout/stderr. The subprocess has no access to this process's
environment variables (e.g. API keys) beyond a minimal PATH.

Calls within one question share a working directory (see reset_workspace) so the
agent can download a large source document once and parse it over several calls;
the directory is reset per question so stale data can never leak into the next
answer.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

TIMEOUT_SECONDS = 45
MAX_OUTPUT_CHARS = 8000
DEFAULT_REQUEST_TIMEOUT = 12  # per-request cap inside the sandbox (see PREAMBLE)

PREAMBLE = textwrap.dedent(
    f"""
    import requests
    import json
    import io
    import re
    import urllib.parse

    # The agent routinely writes requests.get(url) with no timeout=. Without one,
    # requests blocks forever on a slow/hanging government site and the only thing that
    # stops it is this subprocess's wall-clock timeout -- burning the whole budget on a
    # single dead URL. Default every call to a real timeout so a bad host costs seconds,
    # not the entire run; explicit timeout= in agent code still wins.
    _orig_request = requests.Session.request

    def _request_with_default_timeout(self, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = {DEFAULT_REQUEST_TIMEOUT}
        return _orig_request(self, *args, **kwargs)

    requests.Session.request = _request_with_default_timeout

    # Several government sites (mospi.gov.in among them) serve incomplete cert chains, so
    # the agent legitimately falls back to verify=False. urllib3 then prints a multi-KB
    # warning per request, which crowds out real output in the tool result it sees.
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    headers = {{"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}}
    """
)
# pandas/numpy/BeautifulSoup/pdfplumber are intentionally NOT preimported here -- each
# pulls in real memory (pandas+numpy alone is ~150MB+, pdfplumber drags in Pillow), and
# on Render's 512MB free tier, forcing all of them into every subprocess regardless of
# need was causing OOM restarts. The agent already re-imports what it uses per call
# (confirmed from run traces), so import only-what's-needed keeps peak memory down.


def _as_text(stream) -> str:
    """TimeoutExpired carries bytes even when the run was started with text=True."""
    if not stream:
        return ""
    return stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else stream


_work_dir: str | None = None


def reset_workspace() -> str:
    """Starts a fresh working directory for a new question.

    Files persist between run_python calls (so a 50MB PDF is downloaded once and
    parsed over several calls instead of re-fetched every time), but must NOT
    persist between questions -- a stale file from an earlier question could
    otherwise be read as if it were this question's source data.
    """
    global _work_dir
    if _work_dir:
        shutil.rmtree(_work_dir, ignore_errors=True)
    _work_dir = tempfile.mkdtemp(prefix="agent_ws_")
    return _work_dir


def _get_work_dir() -> str:
    """Returns the current workspace, creating one if a run never reset it."""
    if not _work_dir or not os.path.isdir(_work_dir):
        return reset_workspace()
    return _work_dir


def run_python(code: str) -> dict:
    """Execute `code` in a sandboxed subprocess. Returns dict with stdout/stderr/error."""
    workdir = _get_work_dir()
    script_path = Path(workdir) / "_snippet.py"
    script_path.write_text(PREAMBLE + "\n" + code)
    try:
        proc = subprocess.run(
            # -u (unbuffered) so prints reach us as they happen; otherwise a call that times
            # out is killed with its output still sitting in Python's stdout buffer, and the
            # partial-output handling below has nothing to return.
            [sys.executable, "-u", str(script_path)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            },
        )
    except subprocess.TimeoutExpired as exc:
        # Return whatever the script printed before it ran out of time. Discarding it left
        # the agent with no evidence of the progress it had made (e.g. that a large PDF had
        # finished downloading), so it would abandon a nearly-working approach entirely.
        partial_out = _as_text(exc.stdout)[-MAX_OUTPUT_CHARS:]
        partial_err = _as_text(exc.stderr)[-MAX_OUTPUT_CHARS:]
        return {
            "stdout": partial_out,
            "stderr": partial_err,
            "error": (
                f"Timed out after {TIMEOUT_SECONDS}s (output above is partial). Files this "
                "call already wrote are still on disk -- continue from them in the next call "
                "instead of starting over, and process less per call (e.g. a slice of pages)."
            ),
        }

    stdout = proc.stdout[-MAX_OUTPUT_CHARS:]
    stderr = proc.stderr[-MAX_OUTPUT_CHARS:]
    return {
        "stdout": stdout,
        "stderr": stderr,
        "error": None if proc.returncode == 0 else f"Exit code {proc.returncode}",
    }
