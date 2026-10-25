"""Idle-time suggestion cache: grid scoring, ranking, cache freshness, the daemon, privacy, notify, export."""

from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from sloth import cache, conditions, daemon, export, models, notify, suggest, synthetic, tools
from sloth import sessions as sessions_mod
from sloth.config import Config

CFG = Config()
END = dt.date(2026, 10, 18)
CANARY = "CANARY-proj-9b1c"


class CountingRF(RandomForestClassifier):
    calls = 0

    def predict_proba(self, X):
        CountingRF.calls += 1
        return super().predict_proba(X)


@pytest.fixture
def fast_model(monkeypatch):
    CountingRF.calls = 0
    monkeypatch.setattr(
        models, "classifier", lambda *a, categorical_features_indices=None, **k: CountingRF(30, random_state=0)
    )


@pytest.fixture
def work_wins(monkeypatch):
    """Deterministic stand-in: sessions that start with Work become focused 70% vs 20%."""

    def proba(train, target, cfg):
        return np.where(target["starter_family"] == "Work", 0.7, 0.2)

    monkeypatch.setattr(conditions, "predict_proba", proba)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("suggest")
    synthetic.write(d, days=28, end=END, seed=9)
    ev = pd.read_csv(d / "events.csv", keep_default_na=False)
    code = ev["app"] == "Code.exe"
    ev.loc[code, "title"] = f"main.py - {CANARY} - Visual Studio Code"
    ev.to_csv(d / "events.csv", index=False)
    return d


@pytest.fixture(scope="module")
def ctx(data_dir: Path) -> tools.Context:
    return tools.Context(data_dir, CFG, starter_rules=False)


NOW = pd.Timestamp("2026-10-19 08:20", tz="Asia/Kolkata")


# --- grid & ranking ---------------------------------------------------------------------------


def test_grid_scores_every_context_in_one_call(ctx: tools.Context, fast_model) -> None:
    grid = suggest.option_grid(ctx.sessions, CFG, NOW)
    assert CountingRF.calls == 1
    combos = ctx.sessions.groupby(["hour_bucket", "starter_family"]).size()
    usable = set(combos[combos >= conditions.MIN_N].index.get_level_values(0))
    hours = [NOW.floor("h") + pd.Timedelta(hours=i + 1) for i in range(suggest.GRID_HOURS)]
    assert len(grid) == sum(sessions_mod.hour_bucket(h.hour) in usable for h in hours)  # hours with an option
    assert grid[0]["at"].startswith("2026-10-19T09:00")
    for e in grid:
        assert all(o["sessions_seen"] >= conditions.MIN_N for o in e["options"])
        if e["best"]:
            assert e["best"]["gain"] >= conditions.MIN_GAIN


def test_what_if_unchanged_by_refactor(ctx: tools.Context, fast_model) -> None:
    wi = conditions.what_if(ctx.sessions, CFG)
    assert set(wi) == {"usual", "options", "suggestion", "reason"}
    assert all(o["sessions_seen"] >= conditions.MIN_N for o in wi["options"])


def test_habits_outrank_unusual_options_with_equal_gain() -> None:
    opts = [{"gain": 0.3, "feasibility": 0.05}, {"gain": 0.3, "feasibility": 0.6}]
    for o in opts:
        o["score"] = o["gain"] * o["feasibility"]
    assert max(opts, key=lambda o: o["score"])["feasibility"] == 0.6


def test_render_mentions_evidence_and_resume(ctx: tools.Context, fast_model, work_wins) -> None:
    payload = suggest.build(ctx, NOW)
    entry = next(e for e in payload["grid"] if e["best"])
    text = suggest.render(payload, entry, pd.Timestamp(entry["at"]))
    assert "tend to" in text and f"{entry['best']['sessions_seen']} sessions" in text
    assert "Pick up where you left off" in text


# --- cache ------------------------------------------------------------------------------------


def test_cache_roundtrip_and_freshness(tmp_path: Path, ctx: tools.Context, fast_model, work_wins) -> None:
    synthetic.write(tmp_path, days=3, end=END)
    payload = suggest.build(ctx, NOW)
    cache.write(tmp_path, CFG, payload)
    back = cache.read(tmp_path)
    assert back["grid"] == json.loads(json.dumps(payload["grid"], default=str))
    assert cache.is_fresh(back, tmp_path, CFG, NOW + pd.Timedelta(hours=1))
    assert not cache.is_fresh(back, tmp_path, CFG, NOW + pd.Timedelta(hours=CFG.cache_max_age_h + 1))
    assert not cache.is_fresh(back, tmp_path, Config(focus_max_shifts=5), NOW)  # settings changed
    with (tmp_path / "events.csv").open("a", encoding="utf-8") as f:  # new data arrived
        f.write("2026-10-19T09:00:00+05:30,2026-10-19T09:01:00+05:30,60.0,x.exe,,,Uncategorized,Uncategorized\n")
    assert not cache.is_fresh(back, tmp_path, CFG, NOW)
    assert not list(tmp_path.glob("cache/*.tmp"))  # atomic write left no temp file


def test_pick_nearest_entry_or_none() -> None:
    payload = {"grid": [{"at": "2026-10-19T09:00:00+05:30"}, {"at": "2026-10-19T10:00:00+05:30"}]}
    assert suggest.pick(payload, pd.Timestamp("2026-10-19 09:40", tz="Asia/Kolkata"))["at"].startswith("2026-10-19T10")
    assert suggest.pick(payload, pd.Timestamp("2026-10-20 09:00", tz="Asia/Kolkata")) is None
    assert cache.read(Path("does-not-exist")) is None


# --- daemon -------------------------------------------------------------------------------------


class Fake:
    def __init__(self) -> None:
        self.t = pd.Timestamp("2026-10-19 10:00", tz="UTC")
        self.state = ("not-afk", self.t)
        self.builds = 0
        self.toasts: list[str] = []
        self.logs: list[dict] = []
        self.fresh = False
        self.ram_gb: float | None = 8.0
        self.come_back_during_build = False

    def go_afk(self) -> None:
        self.state = ("afk", self.t)

    def come_back(self) -> None:
        self.state = ("not-afk", self.t)

    def tick(self, minutes: float) -> None:
        self.t += pd.Timedelta(minutes=minutes)

    def build(self, should_stop) -> None:
        self.builds += 1
        if self.come_back_during_build:
            self.come_back()
        if should_stop():
            raise daemon.Aborted
        self.fresh = True

    def daemon(self) -> daemon.Daemon:
        return daemon.Daemon(
            cfg=CFG,
            status=lambda: self.state,
            build=self.build,
            is_fresh=lambda now: self.fresh,
            suggestion=lambda now: ("open Work first", {"open_first": "Work"}),
            notify=lambda title, body: self.toasts.append(body),
            log=self.logs.append,
            ram=lambda: self.ram_gb,
            clock=lambda: self.t,
        )


def _run(f: Fake, d: daemon.Daemon, minutes: int) -> list[str]:
    out = []
    for _ in range(minutes):
        f.tick(1)
        out.append(d.step())
    return out


def test_daemon_builds_once_when_idle_then_suggests_on_return() -> None:
    f = Fake()
    d = f.daemon()
    d.step()
    f.go_afk()
    decisions = _run(f, d, 30)
    assert f.builds == 1 and decisions.count("built") == 1
    assert decisions.index("built") == CFG.idle_build_min - 1  # after exactly idle_build_min minutes
    f.come_back()
    assert d.step() == "suggested"
    assert f.toasts == ["open Work first"] and f.logs[0]["open_first"] == "Work"
    assert d.step() == "wait"  # only once per return


def test_daemon_short_break_no_toast_and_no_build() -> None:
    f = Fake()
    d = f.daemon()
    d.step()
    f.go_afk()
    _run(f, d, 5)
    f.come_back()
    assert d.step().startswith("short break") and not f.toasts and f.builds == 0


def test_daemon_skips_build_when_ram_low_or_cache_fresh() -> None:
    f = Fake()
    f.ram_gb = 1.2
    d = f.daemon()
    f.go_afk()
    assert any(x.startswith("skip build") for x in _run(f, d, 15)) and f.builds == 0
    f2 = Fake()
    f2.fresh = True
    d2 = f2.daemon()
    f2.go_afk()
    assert "cache fresh" in _run(f2, d2, 15) and f2.builds == 0


def test_daemon_build_aborts_when_user_returns() -> None:
    f = Fake()
    f.come_back_during_build = True
    d = f.daemon()
    f.go_afk()
    assert "build aborted: you came back" in _run(f, d, 12)
    assert not f.fresh


# --- privacy ------------------------------------------------------------------------------------


def test_project_name_is_local_only(data_dir: Path, ctx: tools.Context, fast_model, monkeypatch, work_wins) -> None:
    pcfg = Config(suggest_projects=True)
    pctx = tools.Context(data_dir, pcfg, starter_rules=False)
    payload = suggest.build(pctx, NOW)
    assert payload["local_only"]["project"] == CANARY
    assert CANARY not in json.dumps({k: v for k, v in payload.items() if k != "local_only"})
    entry = next(e for e in payload["grid"] if e["best"])
    now = pd.Timestamp(entry["at"])
    assert CANARY in suggest.render(payload, entry, now, local=True)
    assert CANARY not in suggest.render(payload, entry, now, local=False)

    sent: list[str] = []

    def fake_chat(client, cfg, messages, use_tools):
        sent.append(json.dumps(messages))
        return {"message": {"content": "Mornings that start with Work tend to go well."}}

    from sloth import agent

    monkeypatch.setattr(agent, "_chat", fake_chat)
    suggest.polish(suggest.facts(payload, entry, now), pctx)
    assert sent and all(CANARY not in m for m in sent)


def test_polish_discards_text_with_invented_numbers(ctx: tools.Context, fast_model, monkeypatch, work_wins) -> None:
    from sloth import agent

    monkeypatch.setattr(agent, "_chat", lambda *a, **k: {"message": {"content": "97% of the time, trust me."}})
    payload = suggest.build(ctx, NOW)
    entry = next(e for e in payload["grid"] if e["best"])
    assert suggest.polish(suggest.facts(payload, entry, NOW), ctx) is None


# --- notify & export ----------------------------------------------------------------------------


def test_windows_toast_command_escapes_text() -> None:
    cmd = notify.command("Sloth & co", "it's <done> $x", platform="win32")
    assert cmd[:2] == ["powershell.exe", "-NoProfile"] and cmd[-2] == "-EncodedCommand"
    script = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert "Sloth &amp; co" in script and "it''s &lt;done&gt; $x" in script


def test_export_since_last_appends_only_newer(tmp_path: Path, monkeypatch) -> None:
    t0 = dt.datetime(2026, 10, 19, 9, 0, tzinfo=dt.UTC)
    old = [
        {"start": t0 + dt.timedelta(hours=h), "end": t0 + dt.timedelta(hours=h, minutes=10), "duration_s": 600.0,
         "app": "a.exe", "title": "", "url": "", "category": "Uncategorized"}
        for h in range(3)
    ]  # fmt: skip
    export.write_events(old, tmp_path / "events.csv")
    fetched_from: list[dt.datetime] = []

    def fake_get(base, path):
        if path == "/api/0/buckets/":
            return {"w": {"id": "w", "type": "currentwindow"}, "a": {"id": "a", "type": "afkstatus"}}
        return {}

    def fake_fetch(base, win_b, afk_b, web_b, start, end):
        fetched_from.append(start)
        ts = (t0 + dt.timedelta(hours=2, minutes=5)).isoformat()  # overlaps the last hour: re-fetched
        new = (t0 + dt.timedelta(hours=4)).isoformat()
        return ([{"timestamp": ts, "duration": 900, "data": {"app": "a.exe", "title": ""}},
                 {"timestamp": new, "duration": 300, "data": {"app": "b.exe", "title": ""}}], [])  # fmt: skip

    monkeypatch.setattr(export, "get", fake_get)
    monkeypatch.setattr(export, "fetch_active_events", fake_fetch)
    export.main(["--since-last", "--out", str(tmp_path)])
    rows = export.read_events(tmp_path / "events.csv")
    assert fetched_from[0] == t0 + dt.timedelta(hours=1, minutes=10)  # last end (2:10) minus 1 h
    assert [r["app"] for r in rows] == ["a.exe", "a.exe", "a.exe", "b.exe"]  # 2 kept + re-fetched + new
    assert np.isclose(rows[2]["duration_s"], 900)
