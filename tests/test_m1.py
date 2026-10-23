"""M1: synthetic data, exporter helpers, block building, config and the network guard."""

from __future__ import annotations

import datetime as dt
import os
import socket

import pandas as pd
import pytest

from sloth import export, features, netguard, synthetic
from sloth.config import Config, load_dotenv

IST = synthetic.IST


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return synthetic.generate(days=28, end=dt.date(2026, 10, 18), seed=1)


# --- synthetic -------------------------------------------------------------------------------


def test_synthetic_is_deterministic() -> None:
    a = synthetic.generate(days=3, end=dt.date(2026, 10, 18), seed=3)
    b = synthetic.generate(days=3, end=dt.date(2026, 10, 18), seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_uncategorized_share_is_about_30_percent(events: pd.DataFrame) -> None:
    share = events.loc[events["category"] == synthetic.UNCATEGORIZED, "duration_s"].sum() / events["duration_s"].sum()
    assert 0.2 <= share <= 0.4


def test_every_true_category_is_learnable(events: pd.DataFrame) -> None:
    """Each hidden category must also appear as a rule-matched category, or no model could learn it."""
    hidden = set(events["true_category"]) - {synthetic.UNCATEGORIZED}
    assert hidden <= set(events["category"])


def test_weekends_differ_from_weekdays(events: pd.DataFrame) -> None:
    day = events["start"].map(lambda t: t.date())
    hours = events.groupby(day)["duration_s"].sum() / 3600
    weekend = hours[[d.weekday() >= 5 for d in hours.index]]
    weekday = hours[[d.weekday() < 5 for d in hours.index]]
    assert weekend.mean() < weekday.mean() - 1


def test_write_produces_exporter_format(tmp_path) -> None:
    synthetic.write(tmp_path, days=2, end=dt.date(2026, 10, 18))
    ev = pd.read_csv(tmp_path / "events.csv")
    bl = pd.read_csv(tmp_path / "blocks.csv")
    assert list(ev.columns) == [*export.EVENT_FIELDS, "true_category"]
    assert list(bl.columns) == export.BLOCK_FIELDS
    assert len(ev) > 0 and len(bl) > 0


# --- exporter --------------------------------------------------------------------------------

RULES = [
    {"name": ["Work"], "rule": {"type": "regex", "regex": "Code|GitHub"}},
    {"name": ["Work", "Programming"], "rule": {"type": "regex", "regex": "Visual Studio Code"}},
    {"name": ["Media"], "rule": {"type": "regex", "regex": "youtube", "ignore_case": True}},
    {"name": ["Broken"], "rule": {"type": "regex", "regex": "("}},
    {"name": ["None"], "rule": {"type": "none"}},
]


def test_categorize_prefers_deepest_match_and_skips_bad_rules() -> None:
    rules = export.compile_rules(RULES)
    assert len(rules) == 3
    assert export.categorize(rules, "Code.exe", "x.py - Visual Studio Code") == "Work > Programming"
    assert export.categorize(rules, "chrome.exe", "", "https://YouTube.com/") == "Media"
    assert export.categorize(rules, "notepad.exe", "todo.txt") == export.UNCAT


def test_load_rules_falls_back_to_web_ui_export(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(export, "get", lambda base, path: {})  # v0.13: server has no classes
    path = tmp_path / "aw-category-export-default.json"
    path.write_text(pd.Series({"id": "default", "categories": RULES}).to_json(), encoding="utf-8")
    rules, source = export.load_rules("http://localhost:5600", path)
    assert len(rules) == 3 and source == str(path)
    assert export.load_rules("http://localhost:5600", tmp_path / "missing.json") == ([], "none found")


def test_aggregate_blocks_splits_events_across_boundaries() -> None:
    t0 = dt.datetime(2026, 10, 1, 9, 20, tzinfo=IST)
    rows = [
        {
            "start": t0,
            "end": t0 + dt.timedelta(minutes=20),
            "duration_s": 1200.0,
            "app": "a",
            "category": "X",
        },
    ]
    blocks = export.aggregate_blocks(rows, 30)
    secs = {k.strftime("%H:%M"): round(v["secs"]) for k, v in blocks.items()}
    assert secs == {"09:00": 600, "09:30": 600}


# --- features ---------------------------------------------------------------------------------


def test_blocks_keep_every_active_minute(events: pd.DataFrame) -> None:
    b = features.build_blocks(events, Config(timezone="Asia/Kolkata"))
    assert b["active_min"].sum() == pytest.approx(events["duration_s"].sum() / 60, rel=1e-6)
    assert len(b) % 48 == 0  # full grid of 30-minute blocks


def test_focus_minutes_need_low_switching() -> None:
    """The original M1-M3 rule, now selected explicitly with focus_rule="switches"."""
    t = dt.datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    rows = []
    for i in range(10):  # 10 alternating 2-minute events: 9 switches
        s = t + dt.timedelta(minutes=2 * i)
        rows.append(
            {"start": s, "duration_s": 120.0, "app": "Code.exe" if i % 2 else "Discord.exe", "title": "",
             "url": "", "category": "Work > Programming" if i % 2 else "Comms > Chat"}
        )  # fmt: skip
    ev = pd.DataFrame(rows)
    block = lambda cfg: features.build_blocks(ev, cfg).query("active_min > 0").iloc[0]  # noqa: E731
    assert block(Config(focus_rule="switches", focus_max_switches=6))["focus_min"] == 0
    loose = block(Config(focus_rule="switches", focus_max_switches=20))
    assert loose["focus_min"] == pytest.approx(10) and loose["n_switches"] == 9


def _block_of(sequence: list[tuple[str, str]], cfg: Config) -> pd.Series:
    """One block from (app, category) events of 2 minutes each, starting 10:00."""
    t = dt.datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    rows = [
        {"start": t + dt.timedelta(minutes=2 * i), "duration_s": 120.0, "app": app, "title": "", "url": "",
         "category": cat}
        for i, (app, cat) in enumerate(sequence)
    ]  # fmt: skip
    return features.build_blocks(pd.DataFrame(rows), cfg).query("active_min > 0").iloc[0]


CODE = ("Code.exe", "Work > Programming")
DOCS = ("chrome.exe", "Work")
VIDEO = ("chrome.exe", "Media > Video")
MYSTERY = ("mystery.exe", "Uncategorized")


def test_focus_shifts_ignore_work_to_work_switching() -> None:
    b = _block_of([CODE, DOCS] * 5, Config(focus_rule="shifts", focus_max_shifts=2))
    assert b["n_switches"] == 9 and b["n_shifts"] == 0
    assert b["focus_min"] == pytest.approx(20)  # all of it counts; the old rule would give 0


def test_focus_shifts_count_work_to_media() -> None:
    seq = [CODE, VIDEO, CODE, VIDEO, CODE, VIDEO, CODE]
    b = _block_of(seq, Config(focus_rule="shifts", focus_max_shifts=2))
    assert b["n_shifts"] == 3 and b["focus_min"] == 0
    assert _block_of(seq, Config(focus_rule="shifts", focus_max_shifts=3))["focus_min"] == pytest.approx(8)


def test_uncategorized_is_neutral() -> None:
    cfg = Config(focus_rule="shifts", focus_max_shifts=0)
    assert _block_of([CODE, MYSTERY, CODE, MYSTERY, CODE], cfg)["n_shifts"] == 0
    assert _block_of([CODE, MYSTERY, VIDEO], cfg)["n_shifts"] == 1  # work -> ? -> video is a shift
    assert features.count_shifts([VIDEO[1], CODE[1]], cfg) == 0  # non-work -> work is not a shift


def test_unknown_focus_rule_fails_loudly() -> None:
    with pytest.raises(ValueError):
        features.passes_focus_rule(0, 0, Config(focus_rule="vibes"))


def test_tokens_drop_stopwords_and_keep_domain() -> None:
    assert features.tokens("main.py - sloth - Visual Studio Code", "https://www.github.com/x") >= {
        "main", "sloth", "visual", "studio", "code", "github",
    }  # fmt: skip
    assert "the" not in features.tokens("the new tab")


# --- config & netguard ------------------------------------------------------------------------


def test_dotenv_does_not_override_real_environment(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nSLOTH_TEST_A=from_file\nSLOTH_TEST_B='quoted'\nSLOTH_TEST_EMPTY=\n")
    monkeypatch.setenv("SLOTH_TEST_A", "from_env")
    monkeypatch.delenv("SLOTH_TEST_B", raising=False)
    load_dotenv(env)
    assert os.environ["SLOTH_TEST_A"] == "from_env"
    assert os.environ["SLOTH_TEST_B"] == "quoted"
    assert "SLOTH_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("SLOTH_TEST_B")


def test_netguard_blocks_non_local_and_allows_loopback() -> None:
    netguard.install()
    try:
        with pytest.raises(netguard.NonLocalNetworkError):
            socket.create_connection(("1.1.1.1", 443), timeout=1)
        with pytest.raises(netguard.NonLocalNetworkError):
            socket.getaddrinfo("example.com", 443)
        socket.getaddrinfo("localhost", 5600)  # must not raise
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            pass
        server.close()
    finally:
        netguard.uninstall()
