"""Settings, read from `sloth.toml` in the working directory when present.

Secrets and machine-specific paths go in `.env` (never committed): TABPFN_TOKEN, SLOTH_DATA_DIR.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    aw_url: str = "http://localhost:5600"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "gemma4:e4b"
    # Gemma 4 defaults to a 128k context; Ollama reserves memory for it. Our prompts are ~2k tokens.
    llm_num_ctx: int = 8192
    # Long enough for the tool rounds of one command, short enough that the next command's TabPFN
    # work doesn't share 16 GB with a resident Gemma (that ran the machine out of memory once).
    llm_keep_alive: str = "2m"
    timezone: str = "Asia/Kolkata"
    block_minutes: int = 30
    # Focus minutes: active minutes in these categories (prefix match on "A > B" paths), counted
    # only in blocks that pass the focus rule:
    # - "shifts" (default, RescueTime-style): at most `focus_max_shifts` shifts from work to a known
    #   non-work category. Work-to-work switching (editor <-> docs) and Uncategorized time are free.
    # - "switches" (the original M1-M3 rule): at most `focus_max_switches` app changes of any kind.
    work_categories: list[str] = field(default_factory=lambda: ["Work"])
    focus_rule: str = "shifts"
    focus_max_shifts: int = 2
    focus_max_switches: int = 6
    # Sessions: a gap this long (AFK-filtered) starts a new session; a session with at least
    # `focus_session_min` focus minutes counts as focused.
    session_gap_min: int = 15
    focus_session_min: int = 25
    # TabPFN on CPU handles about this many training rows comfortably.
    max_train_rows: int = 5000
    tabpfn_version: str = "v3.5-fast"
    seed: int = 0

    @property
    def events_csv(self) -> Path:
        return self.data_dir / "events.csv"

    @property
    def labels_csv(self) -> Path:
        return self.data_dir / "labels.csv"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=value lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def load(path: Path = Path("sloth.toml")) -> Config:
    load_dotenv()
    cfg = Config()
    if os.environ.get("SLOTH_DATA_DIR"):
        cfg.data_dir = Path(os.environ["SLOTH_DATA_DIR"])
    if path.exists():
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        for key, value in raw.items():
            if key not in known:
                raise ValueError(f"Unknown setting {key!r} in {path}")
            default = getattr(cfg, key)
            setattr(cfg, key, Path(value) if isinstance(default, Path) else value)
    return cfg
