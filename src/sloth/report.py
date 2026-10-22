"""`reports/eval.md`: one section per (evaluation, dataset), replaced in place on each run."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

HEADER = """# Evaluation

Numbers written by `sloth eval`. Each section is replaced when its command is re-run.
Real-data sections are computed locally and contain aggregates only.
"""


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Markdown table from dict rows; `columns` is [(key, header)], floats get 3 decimals."""
    out = ["| " + " | ".join(h for _, h in columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k, _ in columns]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def update_section(key: str, body: str, path: Path = Path("reports") / "eval.md") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else HEADER
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"<!-- {key} -->\n{body.strip()}\n\n_Updated {stamp}._\n<!-- /{key} -->"
    pattern = re.compile(rf"<!-- {re.escape(key)} -->.*?<!-- /{re.escape(key)} -->", re.S)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else f"{text.rstrip()}\n\n{block}\n"
    path.write_text(text, encoding="utf-8")
    return path
