"""Sandboxed Python execution tool for the agent.

Runs agent-authored code in a fresh subprocess inside a temp directory, with a
wall-clock timeout and captured stdout/stderr. The subprocess has no access to
this process's environment variables (e.g. API keys) beyond a minimal PATH.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 8000

PREAMBLE = textwrap.dedent(
    """
    import pandas as pd
    import numpy as np
    import requests
    import json
    import io
    import re
    from bs4 import BeautifulSoup
    import pdfplumber
    """
)


def run_python(code: str) -> dict:
    """Execute `code` in a sandboxed subprocess. Returns dict with stdout/stderr/error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "snippet.py"
        script_path.write_text(PREAMBLE + "\n" + code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": os.environ.get("HOME", "/tmp"),
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                },
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "", "error": f"Timed out after {TIMEOUT_SECONDS}s"}

        stdout = proc.stdout[-MAX_OUTPUT_CHARS:]
        stderr = proc.stderr[-MAX_OUTPUT_CHARS:]
        return {
            "stdout": stdout,
            "stderr": stderr,
            "error": None if proc.returncode == 0 else f"Exit code {proc.returncode}",
        }
