"""Label Uncategorized blocks with TabPFN, learning from blocks your category rules already cover.

Unit of work is a 30-minute block. A block's category is the category with the most active time
in it. Blocks whose top category is a real category are training data; blocks whose top
category is Uncategorized get a predicted category and a confidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from sloth import features, models
from sloth.config import Config
from sloth.features import AFK, UNCATEGORIZED
from sloth.timing import timed

OTHER = "Other"
MAX_CLASSES = 10  # TabPFN's native class limit; rarer categories fold into "Other"
NUMERIC = [
    "hour", "weekday", "is_weekend", "block", "active_min", "n_apps", "n_events", "n_switches",
    "browser_share",
]  # fmt: skip
CATEGORICAL = ["top_app", "top_domain", "prev_category", "yday_category", "lastweek_category"]
# Columns that are, or directly reveal, the block's own label. None may be a feature.
FORBIDDEN = {
    "category", "known_category", "true_category", "uncategorized_min", "rule_category",
    "label_source", "title", "url", "token_seconds",
}  # fmt: skip
HAND_LABEL_COLUMNS = ["block_start", "category"]


def load_hand_labels(path: Path | None, cfg: Config) -> pd.Series:
    """`labels.csv` (block_start ISO with offset, category) as a Series keyed by block start."""
    if path is None or not path.exists():
        return pd.Series(dtype=object)
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=object)
    ts = pd.to_datetime(df["block_start"], utc=True, format="ISO8601").dt.tz_convert(cfg.timezone)
    s = pd.Series(df["category"].to_numpy(), index=ts.dt.floor(f"{cfg.block_minutes}min"))
    return s[~s.index.duplicated(keep="last")]


@dataclass
class Matrix:
    vocab: list[str]
    columns: list[str] = field(default_factory=list)
    cat_idx: list[int] = field(default_factory=list)


def make_xy(train: pd.DataFrame, *others: pd.DataFrame, vocab_size: int = 40) -> tuple[Matrix, list[np.ndarray]]:
    """Feature matrices for train and other frames, with categoricals encoded on train."""
    m = Matrix(vocab=features.vocabulary(train, size=vocab_size))
    frames = (train, *others)
    parts = [[f[NUMERIC].to_numpy(dtype=float)] for f in frames]
    m.columns = list(NUMERIC)
    for col in CATEGORICAL:
        encoded = features.encode_categories(*(f[col].astype(str) for f in frames))
        for p, e in zip(parts, encoded, strict=True):
            p.append(e[:, None].astype(float))
        m.cat_idx.append(len(m.columns))
        m.columns.append(col)
    for p, f in zip(parts, frames, strict=True):
        p.append(features.token_features(f, m.vocab).to_numpy())
    m.columns += [f"tok_{i}" for i in range(len(m.vocab))]
    return m, [np.hstack(p) for p in parts]


def leakage_check(columns: list[str]) -> list[str]:
    """Return the problems with a feature list (empty means clean)."""
    problems = [f"{c} is a forbidden column" for c in columns if c in FORBIDDEN]
    allowed = set(NUMERIC) | set(CATEGORICAL)
    problems += [f"{c} is not a known feature" for c in columns if c not in allowed and not c.startswith("tok_")]
    return problems


def _fold_rare(y: pd.Series) -> tuple[pd.Series, list[str]]:
    keep = y.value_counts().index[: MAX_CLASSES - 1].tolist() if y.nunique() > MAX_CLASSES else []
    if not keep:
        return y, sorted(y.unique())
    return y.where(y.isin(keep), OTHER), [*sorted(keep), OTHER]


class Labeller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.seconds = 0.0
        self.columns: list[str] = []

    def fit(self, train: pd.DataFrame, y: pd.Series) -> Labeller:
        train = train.sort_values("block_start").tail(self.cfg.max_train_rows)
        self.y, self.classes = _fold_rare(y.loc[train.index])
        self.train = train
        return self

    def predict(self, blocks: pd.DataFrame) -> pd.DataFrame:
        start = time.perf_counter()
        m, (X_tr, X) = make_xy(self.train, blocks)
        self.columns = m.columns
        if self.y.nunique() < 2:  # TabPFN needs two classes; one class means one answer
            self.seconds = time.perf_counter() - start
            return pd.DataFrame({"predicted": self.y.iloc[0], "confidence": 1.0}, index=blocks.index)
        clf = models.classifier(self.cfg.tabpfn_version, categorical_features_indices=m.cat_idx)
        with timed("tabpfn.classifier.fit", rows=len(X_tr), cols=X_tr.shape[1], classes=len(self.classes)):
            clf.fit(X_tr, self.y.to_numpy())
        if len(X):
            with timed("tabpfn.classifier.predict", rows=len(X)):
                proba = clf.predict_proba(X)
            pred, conf = clf.classes_[proba.argmax(axis=1)], proba.max(axis=1)
        else:
            pred, conf = np.array([], dtype=object), np.array([])
        self.seconds = time.perf_counter() - start
        return pd.DataFrame({"predicted": pred, "confidence": conf}, index=blocks.index)


def split_blocks(blocks: pd.DataFrame, hand: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(training blocks, their labels, blocks to label). Hand labels override and add."""
    active = blocks[blocks["active_min"] > 0]
    y = active["category"].copy()
    if len(hand):
        y = active["block_start"].map(hand).fillna(y)
    known = ~y.isin([UNCATEGORIZED, AFK])
    return active[known], y[known], active[~known]


# --- Evaluation ---------------------------------------------------------------------------


def _baselines(train: pd.DataFrame, y_train: pd.Series, test: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
    out = {}
    t = time.perf_counter()
    majority = y_train.mode().iloc[0]
    out["majority class"] = (np.full(len(test), majority, dtype=object), time.perf_counter() - t)
    t = time.perf_counter()
    by_app = y_train.groupby(train["top_app"]).agg(lambda s: s.mode().iloc[0])
    pred = test["top_app"].map(by_app).fillna(majority).to_numpy()
    out["top_app lookup"] = (pred, time.perf_counter() - t)
    return out


def _compare(lab: Labeller, test: pd.DataFrame, y_true: pd.Series) -> list[dict[str, Any]]:
    y_true = y_true.where(y_true.isin(lab.classes), OTHER)
    preds = {"TabPFN": (lab.predict(test)["predicted"].to_numpy(), lab.seconds)}
    preds.update(_baselines(lab.train, lab.y, test))
    return [
        {
            "method": name,
            "accuracy": float(accuracy_score(y_true, p)),
            "macro_f1": float(f1_score(y_true, p, average="macro", zero_division=0)),
            "n_train": len(lab.train),
            "n_test": len(test),
            "seconds": s,
        }
        for name, (p, s) in preds.items()
    ]


def evaluate_holdout(blocks: pd.DataFrame, cfg: Config, holdout_days: int = 14) -> dict[str, Any]:
    """Train on categorized blocks before the last `holdout_days`, test on categorized blocks after."""
    train_all, y_all, todo = split_blocks(blocks, pd.Series(dtype=object))
    cutoff = blocks.loc[blocks["active_min"] > 0, "date"].max() - pd.Timedelta(days=holdout_days - 1)
    is_test = train_all["date"] >= cutoff
    lab = Labeller(cfg).fit(train_all[~is_test], y_all[~is_test])
    rows = _compare(lab, train_all[is_test], y_all[is_test])
    return {
        "rows": rows,
        "columns": lab.columns,
        "classes": lab.classes,
        "n_uncategorized": len(todo),
        "train_dates": (
            str(train_all.loc[~is_test, "date"].min()),
            str(train_all.loc[~is_test, "date"].max()),
        ),
        "test_dates": (str(cutoff), str(train_all["date"].max())),
    }


def evaluate_truth(blocks: pd.DataFrame, cfg: Config) -> dict[str, Any] | None:
    """Synthetic only: score Uncategorized blocks against the generator's hidden true category."""
    if "true_category" not in blocks.columns:
        return None
    train, y, todo = split_blocks(blocks, pd.Series(dtype=object))
    if todo.empty:
        return None
    lab = Labeller(cfg).fit(train, y)
    return {"rows": _compare(lab, todo, todo["true_category"])}


def evaluate_hand(blocks: pd.DataFrame, cfg: Config, hand: pd.Series) -> dict[str, Any] | None:
    """Score ONLY on hand-tagged blocks; the model never sees those tags."""
    active = blocks[blocks["active_min"] > 0]
    tags = active["block_start"].map(hand) if len(hand) else pd.Series(dtype=object)
    tags = tags.dropna()
    if tags.empty:
        return None
    train, y, _ = split_blocks(blocks.drop(index=tags.index), pd.Series(dtype=object))
    lab = Labeller(cfg).fit(train, y)
    return {"rows": _compare(lab, active.loc[tags.index], tags)}


def label_uncategorized(blocks: pd.DataFrame, cfg: Config, hand: pd.Series | None = None) -> pd.DataFrame:
    """Predicted category and confidence for every Uncategorized block."""
    train, y, todo = split_blocks(blocks, hand if hand is not None else pd.Series(dtype=object))
    pred = Labeller(cfg).fit(train, y).predict(todo)
    return todo[["block_start", "date", "uncategorized_min", "active_min"]].join(pred)
