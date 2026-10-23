"""Sessions: stretches of laptop use separated by a gap of `session_gap_min` or more.

A session row holds only what was known when it started (start time, how it started, what came
before) plus its outcome (focus minutes). That's the unit for "what conditions come before a
focused session", which is answerable even when *when* you focus isn't predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sloth import features
from sloth.config import Config
from sloth.features import AFK, UNCATEGORIZED

HOUR_BUCKETS = [(0, "night 00-06"), (6, "morning 06-12"), (12, "afternoon 12-18"), (18, "evening 18-24")]
START_FEATURES = [
    "start_hour", "weekday", "is_weekend", "minutes_since_prev_session", "hours_since_last_focus",
    "prev_session_focused",
]  # fmt: skip
START_CATEGORICAL = ["hour_bucket", "starter_family", "prev_session_family"]
OUTCOMES = ["focus_min", "length_min", "focused", "end"]
STARTER_WINDOW_MIN = 10


def family(category: str) -> str:
    """Top-level category: "Work > Programming" -> "Work"."""
    return category.split(" > ")[0] if category not in (UNCATEGORIZED, AFK, "") else UNCATEGORIZED


def hour_bucket(hour: int) -> str:
    return [name for start, name in HOUR_BUCKETS if hour >= start][-1]


def build_sessions(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per session. A session's focus minutes apply the block focus rule to the part of
    each block that belongs to the session, so one session's outcome never depends on another's
    events (a block can hold the end of one session and the start of the next)."""
    cols = ["start", "end", "seconds", "app", "category", "block_start"]
    pieces = features._split_at_blocks(events, cfg)[cols].reset_index(drop=True)  # narrow: no titles/URLs
    if pieces.empty:
        return pd.DataFrame(columns=["start", "date", *START_FEATURES, *START_CATEGORICAL, *OUTCOMES])
    pieces["is_work"] = pieces["category"].map(lambda c: features._is_work(c, cfg))
    gap = pieces["start"] - pieces["end"].shift()
    pieces["session"] = (gap >= pd.Timedelta(minutes=cfg.session_gap_min)).cumsum()
    passing = []
    for _, part in pieces.groupby(["session", "block_start"], sort=False):
        apps = part["app"].to_numpy()
        switches = int((apps[1:] != apps[:-1]).sum())
        if features.passes_focus_rule(switches, features.count_shifts(part["category"].to_numpy(), cfg), cfg):
            passing.append(part.index.to_numpy())
    ok = np.zeros(len(pieces), dtype=bool)
    if passing:
        ok[pieces.index.get_indexer(np.concatenate(passing))] = True
    pieces["focus_s"] = np.where(ok & pieces["is_work"].to_numpy(), pieces["seconds"].to_numpy(), 0.0)

    pieces["family"] = pieces["category"].map(family)
    by = pieces.groupby("session", sort=True)
    s = by.agg(start=("start", "first"), end=("end", "last"), length_s=("seconds", "sum"), focus_s=("focus_s", "sum"))
    fam = pieces.groupby(["session", "family"])["seconds"].sum()
    s["main_family"] = fam.loc[fam.groupby(level="session").idxmax()].reset_index(level="family")["family"]
    # How it started: only the first STARTER_WINDOW_MIN minutes, never the rest of the session.
    session_start = pieces["session"].map(s["start"])
    opening = pieces[
        (pieces["start"] < session_start + pd.Timedelta(minutes=STARTER_WINDOW_MIN))
        & (pieces["seconds"] >= 60)
        & ~pieces["category"].isin([UNCATEGORIZED, AFK])
    ]
    s["starter_family"] = opening.groupby("session")["family"].first().reindex(s.index).fillna(UNCATEGORIZED)
    s = s.reset_index(drop=True)
    s["date"] = s["start"].dt.date
    s["start_hour"] = s["start"].dt.hour
    s["weekday"] = s["start"].dt.day_of_week
    s["is_weekend"] = (s["weekday"] >= 5).astype(int)
    s["hour_bucket"] = s["start_hour"].map(hour_bucket)
    s["length_min"] = s.pop("length_s") / 60
    s["focus_min"] = s.pop("focus_s") / 60
    s["focused"] = (s["focus_min"] >= cfg.focus_session_min).astype(int)

    # Context from earlier sessions only.
    s["minutes_since_prev_session"] = (s["start"] - s["end"].shift()).dt.total_seconds() / 60
    s["prev_session_family"] = s["main_family"].shift().fillna("(none)")
    s["prev_session_focused"] = s["focused"].shift()
    last_focus_end = s["end"].where(s["focused"] == 1).ffill().shift()
    s["hours_since_last_focus"] = (s["start"] - last_focus_end).dt.total_seconds() / 3600
    return s.drop(columns=["main_family"])
