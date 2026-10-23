"""M3: forecast plumbing, plan advice and the plan-tomorrow CLI (stand-in regressor)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sloth import features, forecast, models, synthetic
from sloth.config import Config

CFG = Config()
END = dt.date(2026, 10, 18)


class MeanRegressor:
    """Predicts each block's training mean; quantiles are 0 / mean / 2*mean."""

    def fit(self, X, y):
        self.by_block = pd.Series(y).groupby(X[:, 0]).mean()
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        m = pd.Series(X[:, 0]).map(self.by_block).fillna(0).to_numpy()
        if output_type == "main":
            return {"mean": m, "median": m, "mode": m, "quantiles": [m * 0, m, m * 2]}
        return m


@pytest.fixture
def fast_model(monkeypatch):
    monkeypatch.setattr(models, "regressor", lambda *a, **k: MeanRegressor())


@pytest.fixture(scope="module")
def blocks() -> pd.DataFrame:
    return features.build_blocks(synthetic.generate(days=21, end=END, seed=5), CFG)


def test_forecast_day_covers_one_full_day_with_ordered_quantiles(blocks: pd.DataFrame, fast_model) -> None:
    fc = forecast.forecast_day(blocks, CFG)
    assert len(fc) == 48 and fc["date"].nunique() == 1
    assert fc["date"].iloc[0] == END + dt.timedelta(days=1)
    assert (fc["q10"] <= fc["q50"]).all() and (fc["q50"] <= fc["q90"]).all()
    assert fc["pred"].between(0, CFG.block_minutes).all()
    assert fc.attrs["train_dates"][1] == END


def test_extend_to_marks_future_as_unknown(blocks: pd.DataFrame) -> None:
    day = END + dt.timedelta(days=2)
    ext = forecast.extend_to(blocks, day, CFG)
    future = ext[ext["date"] > END]
    assert len(future) == 2 * 48
    assert future["focus_min"].isna().all() and future["active_min"].isna().all()
    # day+2's "yesterday" lag is the unknown day+1, so it must be NaN, not a fake 0
    assert ext.loc[ext["date"] == day, "yday_focus_min"].isna().all()


def test_complete_days_drops_cut_off_final_day(blocks: pd.DataFrame) -> None:
    cut = blocks[blocks["block_start"] < pd.Timestamp(END, tz=CFG.timezone) + pd.Timedelta(hours=11)]
    assert forecast.complete_days(cut)[-1] == END - dt.timedelta(days=1)


def _fc(pred: list[float]) -> pd.DataFrame:
    start = pd.Timestamp("2026-10-24", tz=CFG.timezone)
    p = np.array(pred, dtype=float)
    return pd.DataFrame(
        {
            "block_start": pd.date_range(start, periods=48, freq="30min"),
            "date": dt.date(2026, 10, 24),
            "pred": p,
            "q10": p * 0.5,
            "q90": p * 1.5,
        }
    )


def test_plan_windows_do_not_overlap_and_find_the_peak() -> None:
    pred = [0.0] * 48
    pred[18:21] = [25, 25, 25]  # 09:00-10:30
    pred[30:33] = [15, 15, 15]  # 15:00-16:30
    history = pd.DataFrame({"date": [dt.date(2026, 10, d) for d in range(1, 8)], "focus_min": 120.0})
    p = forecast.plan(_fc(pred), history, CFG)
    assert [w["start"] for w in p["best_windows"]] == ["09:00", "15:00"]
    assert p["best_windows"][0]["expected_focus_min"] == 75
    assert p["predicted_focus_hours"] == pytest.approx(2.0)
    assert not p["rest_day"]


def test_plan_flags_rest_day_when_well_below_typical() -> None:
    pred = [0.0] * 48
    pred[20] = 10
    history = pd.DataFrame({"date": [dt.date(2026, 10, d) for d in range(1, 8)], "focus_min": 180.0})
    assert forecast.plan(_fc(pred), history, CFG)["rest_day"]


def test_backtest_reports_all_methods(blocks: pd.DataFrame, fast_model) -> None:
    res = forecast.backtest(blocks, CFG, days=3)
    assert set(res["block_mae_min"]) == set(forecast.METHODS)
    assert 0 <= res["interval_80_coverage"] <= 1
    assert all({"train_from", "train_to", "actual_h"} <= set(r) for r in res["per_day"])


def test_cli_plan_tomorrow_no_llm(tmp_path: Path, fast_model) -> None:
    from sloth.cli import app

    synthetic.write(tmp_path, days=14, end=END)
    r = CliRunner().invoke(app, ["plan-tomorrow", "--no-llm", "--data", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "Raw forecast for 2026-10-19" in r.output and "Predicted focus" in r.output
