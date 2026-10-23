"""`sloth label --tag`: you hand-tag the blocks TabPFN is least sure about.

This is the one place sloth shows window titles, and only in your own terminal. Nothing here is
logged, written to reports, or passed to a model; your answers go to labels.csv as
(block_start, category) only.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from sloth import features, label, seeds
from sloth.config import Config


def candidates(blocks: pd.DataFrame, cfg: Config, hand: pd.Series) -> pd.DataFrame:
    """Uncategorized, untagged blocks, least confident first."""
    pred = label.label_uncategorized(blocks, cfg, hand)
    if len(hand):
        pred = pred[~pred["block_start"].isin(hand.index)]
    return pred.sort_values("confidence")


def append_label(path: Path, block_start: pd.Timestamp, category: str) -> None:
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        if new:
            f.write(",".join(label.HAND_LABEL_COLUMNS) + "\n")
        f.write(f'{block_start.isoformat()},"{category.replace(chr(34), "")}"\n')


def run(
    data_dir: Path,
    cfg: Config,
    labels_path: Path,
    limit: int = 20,
    starter_rules: bool = True,
    ask: Callable[[str], str] = input,
    console: Console | None = None,
) -> int:
    """Interactive loop. Returns how many blocks were tagged.

    sloth's own guess is deliberately not shown: tags anchored on the model's answer would
    inflate the very accuracy they exist to measure honestly.
    """
    console = console or Console()
    events = features.load_events(data_dir / "events.csv")
    if starter_rules:
        events = seeds.apply(events)
    blocks = features.build_blocks(events, cfg)
    hand = label.load_hand_labels(labels_path, cfg)
    console.print("Finding the blocks sloth is least sure about (one TabPFN run)...")
    todo = candidates(blocks, cfg, hand).head(limit)
    if todo.empty:
        console.print("Nothing left to tag.")
        return 0

    pieces = features._split_at_blocks(events, cfg)
    known = sorted(c for c in blocks["category"].unique() if c not in (features.UNCATEGORIZED, features.AFK))
    known = sorted(set(known) | set(hand.dropna()))
    tagged = 0
    console.print(
        "[dim]Titles below are shown only here, on your screen. Your answer is saved as "
        "(block start, category) only.[/]"
    )
    for i, row in enumerate(todo.itertuples(), 1):
        g = pieces[pieces["block_start"] == row.block_start]
        apps = g.groupby("app")["seconds"].sum().sort_values(ascending=False) / 60
        titles = g.groupby("title")["seconds"].sum().sort_values(ascending=False).head(5)
        t = Table(title=f"[{i}/{len(todo)}] {row.block_start:%a %d %b %H:%M} · {row.active_min:.0f} active min")
        t.add_column("app")
        t.add_column("min", justify="right")
        for app, m in apps.head(4).items():
            t.add_row(escape(str(app)), f"{m:.0f}")
        console.print(t)
        for title in titles.index:
            console.print(f"   · {escape(str(title)[:90])}")
        menu = "  ".join(f"[{n}] {c}" for n, c in enumerate(known, 1))
        console.print(escape(f"   {menu}\n   [Enter]/[s] skip · [q] quit · or type a new category"))
        answer = ask("> ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() in ("s", ""):
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(known):
            category = known[int(answer) - 1]
        else:
            category = answer
            known = sorted({*known, category})
        append_label(labels_path, row.block_start, category)
        tagged += 1
    console.print(f"Tagged {tagged} block(s) → {labels_path}")
    return tagged
