"""Append JSONL logs for model attempt tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_attempt(record: dict, log_path: Path | None = None) -> None:
    path = log_path or Path("reports/attempts_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(record)
    rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
