"""What conditions come before a focused session, and one cheap change to try.

TabPFN learns P(session becomes focused | how and when it started, what came before). That's
evaluated on held-out days against the base rate and the by-hour rate, then used for "what if"
questions, but only about combinations the user has actually done often enough (no extrapolation).
Everything here is correlation from one person's history: suggestions say "tends to", with counts.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sloth import features, models
from sloth.config import Config
from sloth.sessions import HOUR_BUCKETS, START_CATEGORICAL, START_FEATURES
from sloth.timing import timed

MIN_N = 8  # smallest number of sessions behind any shown rate or suggestion
MIN_GAIN = 0.10  # smallest improvement in P(focused) worth suggesting
BUCKET_HOUR = {name: start + 3 for start, name in HOUR_BUCKETS}  # representative hour per bucket


def _matrix(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[list[np.ndarray], list[int]]:
    frames = (train, *others)
    parts = [[f[START_FEATURES].to_numpy(dtype=float)] for f in frames]
    cat_idx = []
    for col in START_CATEGORICAL:
        encoded = features.encode_categories(*(f[col].astype(str) for f in frames))
        for p, e in zip(parts, encoded, strict=True):
            p.append(e[:, None].astype(float))
        cat_idx.append(len(START_FEATURES) + len(cat_idx))
    return [np.hstack(p) for p in parts], cat_idx


def feature_columns() -> list[str]:
    return [*START_FEATURES, *START_CATEGORICAL]


def predict_proba(train: pd.DataFrame, target: pd.DataFrame, cfg: Config) -> np.ndarray:
    """P(focused) for `target` sessions, from a TabPFN fitted on `train` sessions."""
    y = train["focused"].to_numpy()
    if len(np.unique(y)) < 2:
        return np.full(len(target), float(y.mean()) if len(y) else 0.0)
    (X_tr, X), cat_idx = _matrix(train, target)
    clf = models.classifier(cfg.tabpfn_version, categorical_features_indices=cat_idx)
    with timed("tabpfn.conditions.fit", rows=len(X_tr), cols=X_tr.shape[1]):
        clf.fit(X_tr, y)
    with timed("tabpfn.conditions.predict", rows=len(X)):
        proba = clf.predict_proba(X)
    return proba[:, list(clf.classes_).index(1)]


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    both = len(np.unique(y)) == 2
    return {
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def evaluate(sessions: pd.DataFrame, cfg: Config, holdout_days: int = 14) -> dict[str, Any]:
    """Train on sessions before the last `holdout_days`, test on the rest, vs two baselines."""
    cutoff = sessions["date"].max() - pd.Timedelta(days=holdout_days - 1)
    train, test = sessions[sessions["date"] < cutoff], sessions[sessions["date"] >= cutoff]
    y = test["focused"].to_numpy()
    t0 = time.perf_counter()
    preds = {"TabPFN": predict_proba(train, test, cfg)}
    seconds = {"TabPFN": time.perf_counter() - t0}
    base = train["focused"].mean()
    preds["base rate"] = np.full(len(test), base)
    by_hour = train.groupby("hour_bucket")["focused"].mean()
    preds["rate by start hour"] = test["hour_bucket"].map(by_hour).fillna(base).to_numpy()
    rows = [
        {
            "method": m,
            **_scores(y, p),
            "n_train": len(train),
            "n_test": len(test),
            "seconds": seconds.get(m, 0.0),
        }
        for m, p in preds.items()
    ]
    return {
        "rows": rows,
        "train_dates": (str(train["date"].min()), str(train["date"].max())),
        "test_dates": (str(cutoff), str(sessions["date"].max())),
        "focused_rate_train": float(base),
        "focused_rate_test": float(y.mean()) if len(y) else float("nan"),
    }


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """90% Wilson interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def insights(sessions: pd.DataFrame, min_n: int = MIN_N) -> list[dict[str, Any]]:
    """Observed focus rates by one start condition at a time, only where n >= min_n."""
    out = []
    labels = {
        "hour_bucket": "started in the",
        "starter_family": "started with",
        "prev_session_family": "previous session was mostly",
    }
    for col, label in labels.items():
        for value, g in sessions.groupby(col):
            n, k = len(g), int(g["focused"].sum())
            if n < min_n:
                continue
            lo, hi = wilson(k, n)
            out.append(
                {
                    "condition": f"{label} {value}",
                    "sessions": n,
                    "focused": k,
                    "rate": k / n,
                    "low": lo,
                    "high": hi,
                }
            )
    return sorted(out, key=lambda r: -r["rate"])


def what_if(sessions: pd.DataFrame, cfg: Config, min_n: int = MIN_N) -> dict[str, Any]:
    """Compare the user's usual way of starting with alternatives they've done at least min_n times.

    The hypothetical next session: after a night's gap, on the day after the last session.
    """
    combos = sessions.groupby(["hour_bucket", "starter_family"]).size()
    supported = combos[combos >= min_n]
    if supported.empty:
        return {
            "usual": None,
            "options": [],
            "suggestion": None,
            "reason": f"no start pattern seen {min_n}+ times",
        }
    usual = combos.idxmax()
    last = sessions.iloc[-1]
    next_day = pd.Timestamp(last["date"]) + pd.Timedelta(days=1)
    rows = []
    for bucket, fam in supported.index:
        hour = BUCKET_HOUR[bucket]
        start = pd.Timestamp(next_day.date()).tz_localize(sessions["start"].dt.tz) + pd.Timedelta(hours=hour)
        rows.append(
            {
                "start_hour": hour,
                "weekday": next_day.weekday(),
                "is_weekend": int(next_day.weekday() >= 5),
                "minutes_since_prev_session": (start - last["end"]).total_seconds() / 60,
                "hours_since_last_focus": (
                    (start - sessions.loc[sessions["focused"] == 1, "end"].max()).total_seconds() / 3600
                    if sessions["focused"].any()
                    else np.nan
                ),
                "prev_session_focused": float(last["focused"]),
                "hour_bucket": bucket,
                "starter_family": fam,
                "prev_session_family": last["prev_session_family"],
            }
        )
    hypo = pd.DataFrame(rows)
    p = predict_proba(sessions, hypo, cfg)
    options = [
        {
            "hour_bucket": b,
            "starter_family": f,
            "p_focused": float(pi),
            "sessions_seen": int(supported[(b, f)]),
        }
        for (b, f), pi in zip(supported.index, p, strict=True)
    ]
    options.sort(key=lambda o: -o["p_focused"])
    base = next((o for o in options if (o["hour_bucket"], o["starter_family"]) == usual), None)
    best = options[0]
    suggestion = None
    if base and best is not base and best["p_focused"] - base["p_focused"] >= MIN_GAIN:
        suggestion = {
            "change": f"start in the {best['hour_bucket']} with {best['starter_family']} "
            f"instead of the {base['hour_bucket']} with {base['starter_family']}",
            "p_usual": base["p_focused"],
            "p_suggested": best["p_focused"],
            "sessions_seen": best["sessions_seen"],
        }
    return {
        "usual": {"hour_bucket": usual[0], "starter_family": usual[1], "sessions_seen": int(combos[usual])},
        "options": options,
        "suggestion": suggestion,
        "reason": None if suggestion else f"no supported alternative beats your usual start by {MIN_GAIN:.0%}+",
    }


def routine_score(blocks: pd.DataFrame) -> float:
    """Share of focus blocks that also had focus in the same slot yesterday or a week earlier."""
    f = blocks[blocks["focus_min"] > 0]
    if f.empty:
        return 0.0
    return float(((f["yday_focus_min"] > 0) | (f["lastweek_focus_min"] > 0)).mean())


def mode(blocks: pd.DataFrame, threshold: float = 0.4) -> str:
    return "routine" if routine_score(blocks) >= threshold else "design"
