"""The forecast for day D must not depend on anything that happens at or after D 00:00.

Strategy: build features from the real event stream, then again after scrambling every event that
starts at or after D 00:00 (drop some, move some, relabel some, add new ones). Every feature of
day D's blocks, and every feature and target of earlier blocks, must be identical.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from sloth import features, forecast, synthetic
from sloth.config import Config

CFG = Config()
END = dt.date(2026, 10, 18)


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return synthetic.generate(days=28, end=END, seed=11)


def _scramble_from(events: pd.DataFrame, cutoff: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(events["start"], utc=True)
    past, future = events[start < cutoff], events[start >= cutoff].copy()
    future = future.sample(frac=0.6, random_state=seed)  # drop 40%
    future["category"] = rng.permutation(future["category"].to_numpy())
    future["app"] = rng.permutation(future["app"].to_numpy())
    future["duration_s"] = rng.uniform(5, 3000, len(future)).round(1)
    shift = pd.to_timedelta(rng.integers(0, 6 * 3600, len(future)), unit="s")
    future["start"] = pd.to_datetime(future["start"]) + shift  # only ever later, never before cutoff
    extra = future.head(50).assign(category="Work > Programming", app="Code.exe", duration_s=1500.0)
    return pd.concat([past, future, extra], ignore_index=True).sort_values("start", kind="stable")


@pytest.mark.parametrize("days_before_end", [1, 6, 13])
@pytest.mark.parametrize("seed", [0, 1])
def test_day_d_features_ignore_events_from_d_onwards(events: pd.DataFrame, days_before_end: int, seed: int) -> None:
    day = END - dt.timedelta(days=days_before_end)
    cutoff = pd.Timestamp(day, tz=CFG.timezone)
    a = features.build_blocks(events, CFG)
    b = features.build_blocks(_scramble_from(events, cutoff, seed), CFG)

    def rows(df: pd.DataFrame, which: str) -> pd.DataFrame:
        sel = df["date"] == day if which == "D" else df["date"] < day
        return df.loc[sel].set_index("block_start")

    # Sanity: the scramble really changed day D itself...
    assert not np.allclose(rows(a, "D")["focus_min"], rows(b, "D")["focus_min"].reindex(rows(a, "D").index))
    # ...but none of day D's forecast features moved,
    pd.testing.assert_frame_equal(rows(a, "D")[forecast.FEATURES], rows(b, "D")[forecast.FEATURES])
    # and the training history (features + targets before D) is untouched.
    cols = [*forecast.FEATURES, "focus_min"]
    pd.testing.assert_frame_equal(rows(a, "past")[cols], rows(b, "past")[cols])


def test_tomorrow_features_match_backtest_features(events: pd.DataFrame) -> None:
    """`forecast_day` builds tomorrow's rows from history only; they must equal the rows the
    backtest sees for the same day when that day's events do exist."""
    day = END - dt.timedelta(days=3)
    full = features.build_blocks(events, CFG)
    start = pd.to_datetime(events["start"], utc=True)
    history = features.build_blocks(events[start < pd.Timestamp(day, tz=CFG.timezone)], CFG)
    ext = forecast.extend_to(history, day, CFG)
    got = ext[ext["date"] == day].set_index("block_start")[forecast.FEATURES]
    want = full[full["date"] == day].set_index("block_start")[forecast.FEATURES]
    pd.testing.assert_frame_equal(got, want)


def test_backtest_trains_strictly_before_each_test_day(events: pd.DataFrame, monkeypatch) -> None:
    seen: list[tuple[dt.date, dt.date]] = []

    def fake_predict(train: pd.DataFrame, target: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        seen.append((train["date"].max(), target["date"].min()))
        assert target["date"].nunique() == 1
        out = pd.DataFrame({"pred": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0}, index=target.index)
        out.attrs["train_dates"] = (train["date"].min(), train["date"].max())
        return out

    monkeypatch.setattr(forecast, "_predict", fake_predict)
    res = forecast.backtest(features.build_blocks(events, CFG), CFG, days=5)
    assert len(seen) == 5 and len(res["per_day"]) == 5
    assert all(train_max < test_day for train_max, test_day in seen)
    assert all(r["train_to"] < r["date"] for r in res["per_day"])
