"""Builds a JSONL trace per agent run and keeps it published as a public GitHub Gist."""
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

LOCAL_LOG_PATH = Path(__file__).parent / "run.jsonl"
GITHUB_API = "https://api.github.com"


class RunLogger:
    def __init__(self):
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.gist_id = os.environ.get("GIST_ID")
        self.gist_owner = os.environ.get("GITHUB_USERNAME")
        self._session = requests.Session()
        if self.github_token:
            self._session.headers.update(
                {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github+json",
                }
            )

    def _ensure_gist(self) -> str:
        """Creates the gist on first use if GIST_ID isn't set. Returns gist id."""
        if self.gist_id:
            return self.gist_id
        resp = self._session.post(
            f"{GITHUB_API}/gists",
            json={
                "description": "Telegram data-analyst bot run log",
                "public": True,
                "files": {"run.jsonl": {"content": "{}\n"}},
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self.gist_id = data["id"]
        self.gist_owner = data["owner"]["login"]
        logger.warning(
            "Created new gist %s for run logs. Set GIST_ID=%s in your environment so the "
            "log_url stays stable across restarts instead of creating a new gist each time.",
            self.gist_id,
            self.gist_id,
        )
        return self.gist_id

    def log_run(self, record: dict) -> str:
        """Appends `record` as one JSON line locally and pushes the file to the gist.

        Returns the public raw URL for the gist file (log_url).
        """
        record = {"ts": time.time(), **record}
        line = json.dumps(record, ensure_ascii=False)

        with open(LOCAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if not self.github_token:
            # No gist publishing configured; caller must host LOCAL_LOG_PATH some other way.
            return str(LOCAL_LOG_PATH)

        gist_id = self._ensure_gist()
        full_content = LOCAL_LOG_PATH.read_text(encoding="utf-8")
        resp = self._session.patch(
            f"{GITHUB_API}/gists/{gist_id}",
            json={"files": {"run.jsonl": {"content": full_content}}},
            timeout=15,
        )
        resp.raise_for_status()
        if not self.gist_owner:
            self.gist_owner = resp.json()["owner"]["login"]
        # Stable "latest" alias (no revision sha), unlike the per-revision raw_url.
        return f"https://gist.githubusercontent.com/{self.gist_owner}/{gist_id}/raw/run.jsonl"
