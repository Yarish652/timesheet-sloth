"""Forecast tomorrow's focus minutes per block with TabPFNRegressor, and backtest it.

Only information available the evening before is used: calendar position plus lags of at least
one day (same block yesterday, same block last week, 4-week same-weekday mean, yesterday's total).
The previous-block lag is deliberately absent, since for tomorrow it isn't known yet.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import numpy as np
import pandas as pd

from sloth import features, models
from sloth.config import Config
from sloth.timing import timed

FEATURES = [
    "block", "hour", "weekday", "is_weekend", "yday_focus_min", "lastweek_focus_min",
    "focus_4wk_mean", "day_focus_yday", "yday_active_min", "lastweek_active_min",
]  # fmt: skip
QUANTILES = [0.1, 0.5, 0.9]
# "always zero" is there because MAE rewards the median, and most blocks have no focus at all.
BASELINES = {
    "same as last week": "lastweek_focus_min",
    "rolling 4-week average": "focus_4wk_mean",
    "always zero": None,
}
METHODS = ["TabPFN", *BASELINES]
_LAG_COLUMNS = ("prev_", "yday_", "lastweek_", "focus_4wk_mean", "day_focus_yday")


def extend_to(blocks: pd.DataFrame, day: dt.date, cfg: Config) -> pd.DataFrame:
    """Append unknown (NaN) blocks up to and including `day`, then recompute the lag features."""
    last = blocks["block_start"].max()
    end = pd.Timestamp(day, tz=cfg.timezone) + pd.Timedelta(days=1)
    step = pd.Timedelta(minutes=cfg.block_minutes)
    new = pd.date_range(last + step, end, freq=step, inclusive="left")
    if len(new) == 0:
        return blocks
    future = pd.DataFrame(
        {"block_start": new, "active_min": np.nan, "focus_min": np.nan, "category": features.AFK}
    )
    keep = blocks.drop(columns=[c for c in blocks.columns if c.startswith(_LAG_COLUMNS)])
    return features.add_calendar(pd.concat([keep, future], ignore_index=True), cfg)


def complete_days(blocks: pd.DataFrame) -> list[dt.date]:
    """Dates with activity, minus a final day that looks cut off by the export."""
    active = blocks[blocks["active_min"] > 0]
    dates = sorted(active["date"].unique())
    if dates and active["block_start"].max().hour < 23:
        dates = dates[:-1]
    return dates


def _predict(train: pd.DataFrame, target: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    train = train.dropna(subset=["focus_min"]).sort_values("block_start").tail(cfg.max_train_rows)
    reg = models.regressor(cfg.tabpfn_version)
    with timed("tabpfn.regressor.fit", rows=len(train), cols=len(FEATURES)):
        reg.fit(train[FEATURES].to_numpy(dtype=float), train["focus_min"].to_numpy(dtype=float))
    X = target[FEATURES].to_numpy(dtype=float)
    with timed("tabpfn.regressor.predict", rows=len(X), train_rows=len(train)):
        out = reg.predict(X, output_type="main", quantiles=QUANTILES)
    limit = float(cfg.block_minutes)
    res = pd.DataFrame(index=target.index)
    res["pred"] = np.clip(np.asarray(out["mean"]), 0, limit)
    for level, q in zip(QUANTILES, out["quantiles"], strict=True):
        res[f"q{int(level * 100)}"] = np.clip(np.asarray(q), 0, limit)
    res.attrs["train_dates"] = (train["date"].min(), train["date"].max())
    return res


def forecast_day(blocks: pd.DataFrame, cfg: Config, day: dt.date | None = None) -> pd.DataFrame:
    """Per-block forecast for `day` (default: the day after the last complete day)."""
    if day is None:
        days = complete_days(blocks)
        day = days[-1] + dt.timedelta(days=1)
    ext = extend_to(blocks, day, cfg)
    target = ext[ext["date"] == day]
    pred = _predict(ext[ext["date"] < day], target, cfg)
    out = target[["block_start", "date", "block", "hour"]].join(pred)
    out.attrs.update(pred.attrs)
    return out


def backtest(blocks: pd.DataFrame, cfg: Config, days: int = 14) -> dict[str, Any]:
    """Rolling origin: for each of the last `days` complete days, train only on blocks strictly
    before that day's 00:00 and predict its blocks."""
    start = time.perf_counter()
    test_days = complete_days(blocks)[-days:]
    per_day, rows = [], []
    for day in test_days:
        target = blocks[blocks["date"] == day]
        pred = _predict(blocks[blocks["date"] < day], target, cfg)
        frame = pd.DataFrame(
            {
                "date": day,
                "actual": target["focus_min"].to_numpy(),
                "TabPFN": pred["pred"].to_numpy(),
                "q10": pred["q10"].to_numpy(),
                "q90": pred["q90"].to_numpy(),
                **{
                    m: target[col].fillna(0).to_numpy() if col else np.zeros(len(target))
                    for m, col in BASELINES.items()
                },
            }
        )
        rows.append(frame)
        tr_from, tr_to = pred.attrs["train_dates"]
        per_day.append(
            {
                "date": str(day),
                "train_from": str(tr_from),
                "train_to": str(tr_to),
                "actual_h": frame["actual"].sum() / 60,
                **{f"{m} pred_h": frame[m].sum() / 60 for m in METHODS},
                **{f"{m} mae": float((frame[m] - frame["actual"]).abs().mean()) for m in METHODS},
            }
        )
    bt = pd.concat(rows, ignore_index=True)
    daily = bt.groupby("date")[["actual", *METHODS]].sum() / 60
    return {
        "per_day": per_day,
        "block_mae_min": {m: float((bt[m] - bt["actual"]).abs().mean()) for m in METHODS},
        "daily_mae_hours": {m: float((daily[m] - daily["actual"]).abs().mean()) for m in METHODS},
        "interval_80_coverage": float(((bt["actual"] >= bt["q10"]) & (bt["actual"] <= bt["q90"])).mean()),
        "mean_daily_focus_hours": float(daily["actual"].mean()),
        "seconds": time.perf_counter() - start,
    }


def plan(fc: pd.DataFrame, history: pd.DataFrame, cfg: Config, top: int = 3) -> dict[str, Any]:
    """Turn a forecast into advice: best windows for hard work, and whether it's a rest day."""
    total_h = float(fc["pred"].sum() / 60)
    daily = history.groupby("date")["focus_min"].sum() / 60
    typical = float(daily[daily > 0].median()) if (daily > 0).any() else 0.0
    span = max(1, 90 // cfg.block_minutes)  # 90-minute windows
    roll = fc["pred"].rolling(span).sum().shift(-(span - 1))
    windows: list[dict[str, Any]] = []
    taken: list[pd.Timestamp] = []
    for i in roll.dropna().sort_values(ascending=False).index:
        if len(windows) >= top or roll[i] < 1:
            break
        start = fc.loc[i, "block_start"]
        if any(abs((start - t).total_seconds()) < span * cfg.block_minutes * 60 for t in taken):
            continue
        taken.append(start)
        end = start + pd.Timedelta(minutes=cfg.block_minutes * span)
        windows.append(
            {"start": f"{start:%H:%M}", "end": f"{end:%H:%M}", "expected_focus_min": round(float(roll[i]))}
        )
    return {
        "date": str(fc["date"].iloc[0]),
        "predicted_focus_hours": round(total_h, 2),
        # Sum of per-block 10%/90% quantiles: a rough, wider-than-80% range for the day.
        "range_hours": [round(float(fc["q10"].sum() / 60), 2), round(float(fc["q90"].sum() / 60), 2)],
        "typical_focus_hours": round(typical, 2),
        "rest_day": total_h < 0.5 * typical,
        "best_windows": sorted(windows, key=lambda w: w["start"]),
    }
