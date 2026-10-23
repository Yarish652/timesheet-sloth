"""Turn events into 30-minute blocks with model-ready features.

Raw window titles never reach a model. They are reduced to keyword tokens, and a block keeps
only "share of active time whose title or URL contains token t" for a small vocabulary picked
from the training blocks.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from sloth.config import Config

UNCATEGORIZED = "Uncategorized"
AFK = "(away)"  # category of a block with no activity, used by lag features
_TOKEN = re.compile(r"[a-z][a-z0-9+#]{2,}")
_STOP = {
    "the", "and", "for", "with", "you", "your", "new", "tab", "page", "com", "www", "http",
    "https", "html", "exe", "google", "chrome", "microsoft", "edge", "firefox", "mozilla",
    "window", "untitled", "home", "from", "this", "that", "are", "not", "has", "was",
}  # fmt: skip


def tokens(title: str, url: str = "") -> set[str]:
    text = f"{title} {urlparse(url).hostname or ''}".lower()
    return {t for t in _TOKEN.findall(text) if t not in _STOP}


def domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.")


def load_events(path: str | Path) -> pd.DataFrame:
    """Read an exporter `events.csv` (start, end, duration_s, app, title, url, category)."""
    ev = pd.read_csv(path, keep_default_na=False, dtype={"app": str, "title": str, "url": str})
    ev["start"] = pd.to_datetime(ev["start"], utc=True, format="ISO8601")
    return ev


def _split_at_blocks(ev: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Local-time pieces of events, cut at block boundaries."""
    freq = f"{cfg.block_minutes}min"
    start = pd.to_datetime(ev["start"], utc=True, format="ISO8601").dt.tz_convert(cfg.timezone)
    end = start + pd.to_timedelta(ev["duration_s"], unit="s")
    pieces = ev.assign(start=start, end=end)
    out = []
    while len(pieces):
        bstart = pieces["start"].dt.floor(freq)
        bend = bstart + pd.Timedelta(freq)
        cut = pieces["end"] > bend
        head = pieces.assign(block_start=bstart, end=pieces["end"].where(~cut, bend))
        out.append(head)
        pieces = pieces[cut].assign(start=bend[cut])
    df = pd.concat(out, ignore_index=True)
    df["seconds"] = (df["end"] - df["start"]).dt.total_seconds()
    return df[df["seconds"] > 0].sort_values("start", kind="stable").reset_index(drop=True)


def _is_work(category: str, cfg: Config) -> bool:
    return any(category == w or category.startswith(w + " > ") for w in cfg.work_categories)


def count_shifts(categories: list[str] | np.ndarray, cfg: Config) -> int:
    """Shifts from work to a known non-work category, in order. Uncategorized and away time are
    neutral (skipped), so work -> Uncategorized -> video still counts as one shift."""
    shifts, prev_work = 0, None
    for c in categories:
        if c in (UNCATEGORIZED, AFK):
            continue
        work = _is_work(c, cfg)
        if prev_work and not work:
            shifts += 1
        prev_work = work
    return shifts


def passes_focus_rule(n_switches: int, n_shifts: int, cfg: Config) -> bool:
    if cfg.focus_rule == "switches":
        return n_switches <= cfg.focus_max_switches
    if cfg.focus_rule == "shifts":
        return n_shifts <= cfg.focus_max_shifts
    raise ValueError(f"Unknown focus_rule {cfg.focus_rule!r} (use 'shifts' or 'switches')")


def build_blocks(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per (date, block) over every day in range, including idle blocks.

    Keeps a private `token_seconds` column (dict) used by `token_features`; it is derived from
    titles, so it is never printed or sent anywhere.
    """
    p = _split_at_blocks(events, cfg)
    p["is_work"] = p["category"].map(lambda c: _is_work(c, cfg))
    has_truth = "true_category" in p.columns

    rows = []
    for bstart, g in p.groupby("block_start", sort=True):
        secs = g["seconds"].to_numpy()
        cat_s = g.groupby("category")["seconds"].sum()
        app_s = g.groupby("app")["seconds"].sum()
        apps = g["app"].to_numpy()
        switches = int((apps[1:] != apps[:-1]).sum())
        shifts = count_shifts(g["category"].to_numpy(), cfg)
        tok: dict[str, float] = defaultdict(float)
        for title, url, s in zip(g["title"], g["url"], secs, strict=True):
            for t in tokens(str(title), str(url)):
                tok[t] += s
        dom = g[g["url"] != ""].assign(d=lambda x: x["url"].map(domain))
        known = cat_s.drop(UNCATEGORIZED, errors="ignore")
        work_min = g.loc[g["is_work"], "seconds"].sum() / 60
        row = {
            "block_start": bstart,
            "active_min": secs.sum() / 60,
            "category": cat_s.idxmax(),
            "known_category": known.idxmax() if len(known) else AFK,
            "uncategorized_min": cat_s.get(UNCATEGORIZED, 0.0) / 60,
            "top_app": app_s.idxmax(),
            "top_domain": dom.groupby("d")["seconds"].sum().idxmax() if len(dom) else "",
            "n_apps": len(app_s),
            "n_events": len(g),
            "n_switches": switches,
            "n_shifts": shifts,
            "browser_share": float(secs[(g["url"] != "").to_numpy()].sum() / secs.sum()),
            "work_min": work_min,
            "focus_min": work_min if passes_focus_rule(switches, shifts, cfg) else 0.0,
            "token_seconds": dict(tok),
        }
        if has_truth:
            truth = g.groupby("true_category")["seconds"].sum()
            row["true_category"] = truth.idxmax()
        rows.append(row)
    active = pd.DataFrame(rows)

    # Full grid so idle blocks and lags exist.
    tz = cfg.timezone
    first = active["block_start"].min().normalize()
    last = active["block_start"].max().normalize() + pd.Timedelta(days=1)
    grid = pd.DataFrame(
        {"block_start": pd.date_range(first, last, freq=f"{cfg.block_minutes}min", tz=tz, inclusive="left")}
    )
    b = grid.merge(active, on="block_start", how="left")
    fill = {
        "active_min": 0.0, "uncategorized_min": 0.0, "n_apps": 0, "n_events": 0, "n_switches": 0,
        "n_shifts": 0, "browser_share": 0.0, "work_min": 0.0, "focus_min": 0.0, "category": AFK,
        "known_category": AFK, "top_app": "", "top_domain": "",
    }  # fmt: skip
    b = b.fillna(fill)
    b["token_seconds"] = b["token_seconds"].map(lambda x: x if isinstance(x, dict) else {})
    for c in ("n_apps", "n_events", "n_switches", "n_shifts"):
        b[c] = b[c].astype(int)
    return add_calendar(b, cfg)


def add_calendar(b: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    per_day = 24 * 60 // cfg.block_minutes
    b = b.copy()
    b["date"] = b["block_start"].dt.date
    b["block"] = (b["block_start"].dt.hour * 60 + b["block_start"].dt.minute) // cfg.block_minutes
    b["hour"] = b["block_start"].dt.hour
    b["weekday"] = b["block_start"].dt.day_of_week
    b["is_weekend"] = (b["weekday"] >= 5).astype(int)

    # Lags on the regular grid: previous block, same block yesterday, same block last week.
    for name, shift in (("prev", 1), ("yday", per_day), ("lastweek", 7 * per_day)):
        b[f"{name}_category"] = b["category"].shift(shift).fillna(AFK)
        b[f"{name}_active_min"] = b["active_min"].shift(shift)
        b[f"{name}_focus_min"] = b["focus_min"].shift(shift)
    # Mean focus in this block over the previous 4 same-weekdays.
    weekly = [b["focus_min"].shift(k * 7 * per_day) for k in range(1, 5)]
    b["focus_4wk_mean"] = pd.concat(weekly, axis=1).mean(axis=1)
    b["day_focus_yday"] = b.groupby("date")["focus_min"].transform("sum").shift(per_day)
    return b


def vocabulary(blocks: pd.DataFrame, size: int = 40, min_blocks: int = 5) -> list[str]:
    """Most common tokens by number of active blocks they appear in."""
    df: Counter[str] = Counter()
    for toks in blocks["token_seconds"]:
        df.update(toks.keys())
    return [t for t, n in df.most_common(size) if n >= min_blocks]


def token_features(blocks: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    """Share of each block's active time whose title/URL contains each vocab token."""
    active_s = (blocks["active_min"] * 60).replace(0, np.nan).to_numpy()
    mat = np.zeros((len(blocks), len(vocab)))
    idx = {t: i for i, t in enumerate(vocab)}
    for r, toks in enumerate(blocks["token_seconds"]):
        for t, s in toks.items():
            if t in idx:
                mat[r, idx[t]] = s
    mat = np.nan_to_num(mat / active_s[:, None])
    return pd.DataFrame(mat, columns=[f"tok_{i}" for i in range(len(vocab))], index=blocks.index)


def encode_categories(train: pd.Series, *others: pd.Series) -> list[np.ndarray]:
    """Integer-encode a string column with codes learned on `train` (unseen values share one extra code)."""
    codes = {v: i for i, v in enumerate(sorted(train.unique()))}
    unseen = len(codes)
    return [s.map(lambda v: codes.get(v, unseen)).to_numpy() for s in (train, *others)]
