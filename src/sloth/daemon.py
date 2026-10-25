"""`sloth daemon`: build the suggestion cache while you're away, show a suggestion when you're back.

The decision logic is `Daemon.step`, a small state machine with the clock, ActivityWatch, the
RAM check, the builder and the notifier all injected, so it can be tested without waiting
real minutes.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sloth import export
from sloth.config import Config


class AFKClient:
    """Current AFK status from ActivityWatch's afkstatus bucket (localhost only)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self._bucket: str | None = None

    def status(self) -> tuple[str, pd.Timestamp]:
        """('afk' | 'not-afk', since when)."""
        if self._bucket is None:
            self._bucket = export.pick_bucket(export.get(self.base, "/api/0/buckets/"), "afkstatus")
        ev = export.get(self.base, f"/api/0/buckets/{self._bucket}/events?limit=1")[0]
        return ev["data"]["status"], pd.Timestamp(ev["timestamp"])


def free_ram_gb() -> float | None:
    """Free physical memory, or None if unknown (non-Windows: no check, no dependency)."""
    if sys.platform != "win32":
        return None

    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]  # fmt: skip

    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
    return m.ullAvailPhys / 1024**3


def lower_priority() -> None:
    if sys.platform == "win32":
        below_normal = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)  # type: ignore[attr-defined]
    else:
        os.nice(10)


def unload_llm(cfg: Config) -> None:
    """Ask Ollama to drop the model from RAM right away (best effort)."""
    body = json.dumps({"model": cfg.llm_model, "keep_alive": 0}).encode()
    req = urllib.request.Request(f"{cfg.ollama_url}/api/generate", data=body, method="POST")
    with contextlib.suppress(OSError):
        urllib.request.urlopen(req, timeout=30).close()


class Aborted(Exception):
    """The user came back mid-build."""


@dataclass
class Daemon:
    cfg: Config
    status: Callable[[], tuple[str, pd.Timestamp]]
    build: Callable[[Callable[[], bool]], None]  # gets a "should I stop?" callback
    is_fresh: Callable[[pd.Timestamp], bool]
    suggestion: Callable[[pd.Timestamp], tuple[str, dict[str, Any]] | None]
    notify: Callable[[str, str], Any]
    log: Callable[[dict[str, Any]], None]
    ram: Callable[[], float | None] = free_ram_gb
    clock: Callable[[], pd.Timestamp] = field(default=lambda: pd.Timestamp.now(tz="UTC"))
    say: Callable[[str], None] = lambda _msg: None
    _last_status: str | None = None
    _away_since: pd.Timestamp | None = None
    _built_this_break: bool = False

    def step(self) -> str:
        """One poll. Returns what it decided, for logs and tests."""
        status, since = self.status()
        now = self.clock()
        decision = "wait"
        if status == "afk":
            if self._last_status != "afk":
                self._away_since, self._built_this_break = since, False
            away_min = (now - (self._away_since or since)).total_seconds() / 60
            if away_min >= self.cfg.idle_build_min and not self._built_this_break:
                ram = self.ram()
                if self.is_fresh(now):
                    decision, self._built_this_break = "cache fresh", True
                elif ram is not None and ram < self.cfg.min_free_ram_gb:
                    decision = f"skip build: {ram:.1f} GB free"
                else:
                    try:
                        self.build(lambda: self.status()[0] != "afk")
                        decision = "built"
                    except Aborted:
                        decision = "build aborted: you came back"
                    self._built_this_break = True
        elif status == "not-afk" and self._last_status == "afk" and self._away_since is not None:
            away_min = (now - self._away_since).total_seconds() / 60
            if away_min >= self.cfg.session_gap_min:
                picked = self.suggestion(now)
                if picked:
                    text, record = picked
                    self.notify("Timesheet Sloth", text)
                    self.log({"shown_at": now.isoformat(), "away_min": round(away_min), **record})
                    decision = "suggested"
                else:
                    decision = "no suggestion cached"
            else:
                decision = f"short break ({away_min:.0f} min): no suggestion"
            self._away_since = None
        self._last_status = status
        self.say(f"{now.tz_convert(self.cfg.timezone):%H:%M} {status}: {decision}")
        return decision


def lock(data_dir: Path) -> Path:
    """Single instance: refuse to start if another daemon's PID is alive."""
    p = data_dir / "cache" / "daemon.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            pid = int(p.read_text().strip())
            if pid != os.getpid() and _alive(pid):
                raise RuntimeError(f"sloth daemon already running (pid {pid}); delete {p} if it isn't")
        except ValueError:
            pass
    p.write_text(str(os.getpid()))
    return p


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_forever(d: Daemon, poll_s: int) -> None:
    while True:
        try:
            d.step()
        except OSError as e:  # ActivityWatch down, network blip on localhost, ...
            d.say(f"poll failed: {e}")
        time.sleep(poll_s)
