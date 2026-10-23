"""M4: sessions, focus conditions, tools, the privacy guard and the Gemma agent loop."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from sloth import agent, conditions, features, models, privacy, sessions, synthetic, tools
from sloth.config import Config

CFG = Config()
END = dt.date(2026, 10, 18)
CANARY = "SECRET-CANARY-7f3a"


@pytest.fixture
def fast_model(monkeypatch):
    monkeypatch.setattr(
        models,
        "classifier",
        lambda *a, categorical_features_indices=None, **k: RandomForestClassifier(50, random_state=0),
    )

    class MeanReg:
        def fit(self, X, y):
            self.m = float(np.mean(y))
            return self

        def predict(self, X, output_type="mean", quantiles=None):
            m = np.full(len(X), self.m)
            return {"mean": m, "median": m, "mode": m, "quantiles": [m * 0, m, m * 2]} if output_type == "main" else m

    monkeypatch.setattr(models, "regressor", lambda *a, **k: MeanReg())


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Synthetic data with a canary planted in titles and URLs."""
    d = tmp_path_factory.mktemp("canary")
    synthetic.write(d, days=28, end=END, seed=3)
    ev = pd.read_csv(d / "events.csv", keep_default_na=False)
    idx = ev.index[::7]
    ev.loc[idx, "title"] = ev.loc[idx, "title"] + f" - {CANARY} private chat"
    ev.loc[idx[::2], "url"] = f"https://{CANARY.lower()}.example.com/inbox"
    ev.to_csv(d / "events.csv", index=False)
    return d


@pytest.fixture(scope="module")
def ctx(data_dir: Path) -> tools.Context:
    return tools.Context(data_dir, CFG, starter_rules=False)


# --- sessions ---------------------------------------------------------------------------------


def _ev(spec: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    """(start minute after 09:00, minutes, app, category) -> events."""
    t = dt.datetime(2026, 10, 1, 9, 0, tzinfo=synthetic.IST)
    return pd.DataFrame(
        [
            {
                "start": t + dt.timedelta(minutes=m),
                "duration_s": 60.0 * n,
                "app": a,
                "title": "",
                "url": "",
                "category": c,
            }
            for m, n, a, c in spec
        ]
    )


def test_sessions_split_at_gap_and_carry_previous_context() -> None:
    ev = _ev([(0, 30, "Code.exe", "Work > Programming"), (40, 20, "x.exe", "Media > Video"),
              (90, 10, "x.exe", "Media > Video"), (100, 30, "Code.exe", "Work > Programming")])  # fmt: skip
    s = sessions.build_sessions(ev, CFG)
    assert len(s) == 2  # 30->40 is a 10-min gap (same session); 60->90 is 30 min (new session)
    first, second = s.iloc[0], s.iloc[1]
    assert first["starter_family"] == "Work" and first["focused"] == 1
    assert first["prev_session_family"] == "(none)" and np.isnan(first["minutes_since_prev_session"])
    assert second["starter_family"] == "Media" and second["prev_session_family"] == "Work"
    assert second["minutes_since_prev_session"] == pytest.approx(30)
    assert second["prev_session_focused"] == 1


def test_session_start_features_ignore_the_session_itself(data_dir: Path) -> None:
    """Scrambling everything from a session's start onwards must not move its start features."""
    ev = features.load_events(data_dir / "events.csv")
    s = sessions.build_sessions(ev, CFG)
    target = s.iloc[len(s) // 2]
    cut = target["start"]
    start = pd.to_datetime(ev["start"], utc=True)
    later = ev[start >= cut].copy()
    later["category"] = np.random.default_rng(0).permutation(later["category"].to_numpy())
    later["duration_s"] = later["duration_s"] * 0.5
    s2 = sessions.build_sessions(pd.concat([ev[start < cut], later]), CFG)
    row2 = s2[s2["start"] == target["start"]].iloc[0]
    for col in ["start_hour", "weekday", "minutes_since_prev_session", "hours_since_last_focus",
                "prev_session_focused", "hour_bucket", "prev_session_family"]:  # fmt: skip
        assert row2[col] == target[col] or (pd.isna(row2[col]) and pd.isna(target[col])), col


def test_starter_family_only_sees_the_opening_minutes() -> None:
    """Regression: starter_family once came from the first categorized event *anywhere* in the
    session, so 'started with Uncategorized' meant 'no categorized time at all' and was 0% focused
    by construction."""
    ev = _ev([(0, 12, "mystery.exe", "Uncategorized"), (12, 60, "Code.exe", "Work > Programming")])
    s = sessions.build_sessions(ev, CFG)
    assert len(s) == 1
    assert s.iloc[0]["starter_family"] == "Uncategorized"  # nothing categorized in the first 10 min
    assert s.iloc[0]["focused"] == 1  # ...and the session can still be focused


def test_starter_family_ignores_events_after_the_opening(data_dir: Path) -> None:
    ev = features.load_events(data_dir / "events.csv")
    s = sessions.build_sessions(ev, CFG)
    start = pd.to_datetime(ev["start"], utc=True)
    for target in s.iloc[:: max(1, len(s) // 10)].itertuples():
        cut = target.start + pd.Timedelta(minutes=sessions.STARTER_WINDOW_MIN)
        later = (start >= cut) & (start < target.end)
        scrambled = ev.copy()
        scrambled.loc[later, "category"] = "Media > Video"
        row = sessions.build_sessions(scrambled, CFG).set_index("start").loc[target.start]
        assert row["starter_family"] == target.starter_family


# --- conditions -------------------------------------------------------------------------------


def test_conditions_eval_reports_all_methods(ctx: tools.Context, fast_model) -> None:
    res = conditions.evaluate(ctx.sessions, CFG, holdout_days=7)
    assert [r["method"] for r in res["rows"]] == ["TabPFN", "base rate", "rate by start hour"]
    assert all(0 <= r["brier"] <= 1 for r in res["rows"])


def test_insights_hide_small_groups(ctx: tools.Context) -> None:
    ins = conditions.insights(ctx.sessions, min_n=8)
    assert ins and all(r["sessions"] >= 8 for r in ins)
    assert all(r["low"] <= r["rate"] <= r["high"] for r in ins)


def test_what_if_only_suggests_supported_options(ctx: tools.Context, fast_model) -> None:
    wi = conditions.what_if(ctx.sessions, CFG)
    assert all(o["sessions_seen"] >= conditions.MIN_N for o in wi["options"])
    if wi["suggestion"]:
        assert wi["suggestion"]["p_suggested"] - wi["suggestion"]["p_usual"] >= conditions.MIN_GAIN


def test_wilson_interval_bounds() -> None:
    lo, hi = conditions.wilson(0, 10)
    assert lo == 0 and 0 < hi < 0.3
    lo, hi = conditions.wilson(10, 10)
    assert hi == 1 and lo > 0.7


# --- tools & privacy --------------------------------------------------------------------------


def test_no_tool_ever_returns_the_canary(ctx: tools.Context, fast_model) -> None:
    for name in tools.TOOLS:
        result = tools.call(ctx, name, {"date": str(END), "range": "week"})
        text = json.dumps(result, default=str).lower()
        assert CANARY.lower() not in text, name
        assert "http" not in text and ".com" not in text, name
        privacy.Guard(ctx.events).check(result)  # and the guard agrees


def test_guard_blocks_titles_and_urls_without_echoing_them(ctx: tools.Context) -> None:
    guard = privacy.Guard(ctx.events)
    title = ctx.events.loc[ctx.events["title"].str.contains(CANARY), "title"].iloc[0]
    with pytest.raises(privacy.PrivacyError) as e:
        guard.check({"oops": title})
    assert CANARY not in str(e.value)
    with pytest.raises(privacy.PrivacyError):
        guard.check({"x": f"see {CANARY.lower()}.example.com"})


def test_guard_allows_app_and_category_names(ctx: tools.Context) -> None:
    guard = privacy.Guard(ctx.events)
    guard.check({"apps": list(ctx.events["app"].unique()), "cats": list(ctx.events["category"].unique())})


def test_day_summary_adds_up(ctx: tools.Context) -> None:
    r = tools.day_summary(ctx, str(END))
    assert sum(r["hours_by_category_after_sloth"].values()) == pytest.approx(r["active_hours"], abs=0.05)
    assert r["focus_hours"] <= r["active_hours"]


def test_slow_tools_are_cached_after_prefetch(ctx: tools.Context, fast_model, monkeypatch) -> None:
    tools.prefetch(ctx, ["focus_conditions"])
    monkeypatch.setattr(tools.conditions, "what_if", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    assert "sessions" in tools.call(ctx, "focus_conditions")  # served from the cache, no TabPFN call


def test_unknown_tool_and_bad_args_return_errors(ctx: tools.Context) -> None:
    assert "error" in tools.call(ctx, "rm_rf", {})
    assert "error" in tools.call(ctx, "day_summary", {"date": "not-a-date"})


# --- agent --------------------------------------------------------------------------------------


def _fake_ollama(script: list[dict]) -> tuple[httpx.Client, list[dict]]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=script[len(seen) - 1])

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_agent_dispatches_tools_and_checks_numbers(ctx: tools.Context) -> None:
    day = tools.day_summary(ctx, str(END))
    script = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "day_summary", "arguments": {"date": str(END)}}}],
            }
        },
        {"message": {"role": "assistant", "content": f"Active {day['active_hours']} h, and 99.9 h of magic."}},
    ]
    client, seen = _fake_ollama(script)
    calls = []
    res = agent.run("standup please", ctx, on_call=calls.append, client=client)
    assert [c.name for c in calls] == ["day_summary"]
    assert seen[0]["think"] is False and seen[0]["tools"]
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and CANARY.lower() not in tool_msg["content"].lower()
    assert res.unmatched_numbers == ["99.9"]  # the invented number is flagged, the real one isn't


def test_number_checker_accepts_rounding_percent_and_times() -> None:
    results = [{"focus_hours": 1.234, "rate": 0.375, "start": "09:30", "n": 12}]
    assert agent.check_numbers("1.2 h, 1.23 h, 38%, 37.5%, from 09:30, 12 sessions", results) == []
    assert agent.check_numbers("2.5 h at 10:00", results) == ["10:00", "2.5"]
    assert agent.check_numbers("on 2026-10-18", results) == []  # dates are ignored


def test_number_checker_accepts_either_way_of_rounding_halves() -> None:
    """Regression from the first real timesheet: tool said 5.25, Gemma wrote 5.3 (half-up)."""
    results = [{"uncategorized_hours": 5.25}]
    assert agent.check_numbers("Uncategorized: 5.3 h (or 5.2, or 5)", results) == []
    assert agent.check_numbers("Uncategorized: 5.4 h", results) == ["5.4"]


def test_banner_local_vs_remote() -> None:
    assert agent.banner(Config(), "data").startswith("LLM: LOCAL (gemma4:e4b) · TabPFN: LOCAL")
    assert "REMOTE" in agent.banner(Config(ollama_url="https://api.example.com"), "data")
