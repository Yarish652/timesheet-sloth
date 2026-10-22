"""Wall-clock timing for every TabPFN and LLM call, appended to `reports/timings.jsonl`."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOG_PATH = Path("reports") / "timings.jsonl"


@contextmanager
def timed(what: str, **meta: Any) -> Iterator[dict[str, Any]]:
    """Time a block and log it. Callers may add fields (e.g. token counts) to the yielded dict."""
    record: dict[str, Any] = {"what": what, **meta}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        record["at"] = dt.datetime.now().isoformat(timespec="seconds")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_log(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
