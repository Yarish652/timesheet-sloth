"""The suggestion cache: built while the laptop is idle, read in milliseconds when you sit down."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from sloth.config import Config


def cache_dir(data_dir: Path) -> Path:
    return data_dir / "cache"


def path(data_dir: Path) -> Path:
    return cache_dir(data_dir) / "suggestions.json"


def fingerprint(data_dir: Path, cfg: Config) -> str:
    """Changes when the export changes or any setting changes."""
    ev = data_dir / "events.csv"
    st = ev.stat() if ev.exists() else None
    settings = json.dumps(asdict(cfg), default=str, sort_keys=True)
    raw = f"{st.st_size if st else 0}:{st.st_mtime_ns if st else 0}:{settings}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def write(data_dir: Path, cfg: Config, payload: dict[str, Any]) -> Path:
    """Atomic: a reader never sees a half-written file."""
    out = path(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**payload, "fingerprint": fingerprint(data_dir, cfg)}, default=str), encoding="utf-8")
    os.replace(tmp, out)
    return out


def read(data_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path(data_dir).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_fresh(payload: dict[str, Any] | None, data_dir: Path, cfg: Config, now: pd.Timestamp) -> bool:
    if not payload:
        return False
    age_h = (now - pd.Timestamp(payload["built_at"])).total_seconds() / 3600
    return payload.get("fingerprint") == fingerprint(data_dir, cfg) and 0 <= age_h < cfg.cache_max_age_h


def log_shown(data_dir: Path, record: dict[str, Any]) -> None:
    """What was suggested and when, for the later accept/ignore analysis. No titles, no project."""
    out = cache_dir(data_dir) / "shown.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
