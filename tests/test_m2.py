"""M2: starter rules, labeller plumbing, leakage check, hand labels, tagging and reports."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from typer.testing import CliRunner

from sloth import features, label, models, report, seeds, synthetic, tagging
from sloth.config import Config

CFG = Config()


@pytest.fixture(scope="module")
def blocks() -> pd.DataFrame:
    ev = synthetic.generate(days=21, end=dt.date(2026, 10, 18), seed=2)
    return features.build_blocks(ev, CFG)


@pytest.fixture
def fast_model(monkeypatch):
    """Stand-in classifier so plumbing tests don't spend ~10 s per TabPFN fit."""
    monkeypatch.setattr(
        models,
        "classifier",
        lambda *a, categorical_features_indices=None, **k: RandomForestClassifier(50, random_state=0),
    )


# --- starter rules --------------------------------------------------------------------------


def test_starter_rules_only_fill_uncategorized() -> None:
    ev = pd.DataFrame(
        {
            "app": ["Code.exe", "Code.exe", "chrome.exe", "mystery.exe"],
            "title": ["a.py - Visual Studio Code", "a.py - Visual Studio Code", "Home", "x"],
            "url": ["", "", "https://www.youtube.com/", ""],
            "category": ["Media > Video", "Uncategorized", "Uncategorized", "Uncategorized"],
        }
    )
    out = seeds.apply(ev)
    assert out["category"].tolist() == [
        "Media > Video",
        "Work > Programming",
        "Media > Video",
        "Uncategorized",
    ]
    assert out["label_source"].tolist() == ["your rules", "starter rules", "starter rules", "none"]
    assert out["rule_category"].tolist() == ev["category"].tolist()


# --- leakage ----------------------------------------------------------------------------------


def test_feature_matrix_is_numeric_and_passes_leakage_check(blocks: pd.DataFrame) -> None:
    train, _y, todo = label.split_blocks(blocks, pd.Series(dtype=object))
    m, (X_tr, X) = label.make_xy(train, todo)
    assert X_tr.dtype == np.float64 and X.shape[1] == len(m.columns)
    assert label.leakage_check(m.columns) == []
    assert all(not c.startswith("tok_") or c[4:].isdigit() for c in m.columns)  # no words in names


@pytest.mark.parametrize("bad", ["category", "known_category", "true_category", "title", "url", "rule_category"])
def test_leakage_check_flags_label_columns(bad: str) -> None:
    assert label.leakage_check([*label.NUMERIC, bad])


def test_leakage_check_flags_unknown_columns() -> None:
    assert label.leakage_check(["mystery_feature"])


# --- hand labels --------------------------------------------------------------------------------


def test_hand_labels_parse_offsets_and_keep_last(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    p.write_text(
        "block_start,category\n"
        "2026-10-01T09:00:00+05:30,Work\n"
        "2026-10-01T03:30:00Z,Comms > IM\n"  # same block as above, in UTC: later row wins
        "2026-10-01T10:10:00+05:30,Media > Video\n",  # floors to 10:00
        encoding="utf-8",
    )
    s = label.load_hand_labels(p, CFG)
    assert len(s) == 2
    assert s[pd.Timestamp("2026-10-01 09:00", tz="Asia/Kolkata")] == "Comms > IM"
    assert s[pd.Timestamp("2026-10-01 10:00", tz="Asia/Kolkata")] == "Media > Video"


def test_hand_labels_missing_or_empty(tmp_path: Path) -> None:
    assert label.load_hand_labels(tmp_path / "nope.csv", CFG).empty
    (tmp_path / "labels.csv").write_text("block_start,category\n", encoding="utf-8")
    assert label.load_hand_labels(tmp_path / "labels.csv", CFG).empty


def test_hand_eval_scores_only_tagged_blocks(blocks: pd.DataFrame, fast_model) -> None:
    assert label.evaluate_hand(blocks, CFG, pd.Series(dtype=object)) is None
    todo = blocks[(blocks["active_min"] > 0) & (blocks["category"] == features.UNCATEGORIZED)].head(7)
    hand = pd.Series(todo["true_category"].to_numpy(), index=todo["block_start"].to_numpy())
    res = label.evaluate_hand(blocks, CFG, hand)
    assert res is not None
    assert {r["n_test"] for r in res["rows"]} == {7}
    assert [r["method"] for r in res["rows"]] == ["TabPFN", "majority class", "top_app lookup"]


def test_hand_labels_override_training_labels(blocks: pd.DataFrame) -> None:
    b = blocks[(blocks["active_min"] > 0) & (blocks["category"] == features.UNCATEGORIZED)].iloc[0]
    hand = pd.Series(["Work > Programming"], index=[b["block_start"]])
    _train, y, todo = label.split_blocks(blocks, hand)
    assert y.loc[b.name] == "Work > Programming"
    assert b.name not in todo.index


# --- labeller ---------------------------------------------------------------------------------


def test_label_uncategorized_returns_confidence(blocks: pd.DataFrame, fast_model) -> None:
    pred = label.label_uncategorized(blocks, CFG)
    assert len(pred) == ((blocks["active_min"] > 0) & (blocks["category"] == features.UNCATEGORIZED)).sum()
    assert pred["confidence"].between(0, 1).all()
    assert features.UNCATEGORIZED not in set(pred["predicted"])


def test_rare_classes_fold_into_other() -> None:
    y = pd.Series([f"c{i}" for i in range(12) for _ in range(12 - i)])
    folded, classes = label._fold_rare(y)
    assert len(classes) == label.MAX_CLASSES and classes[-1] == label.OTHER
    assert folded.nunique() == label.MAX_CLASSES


@pytest.mark.slow
def test_real_tabpfn_beats_majority_on_synthetic_holdout(blocks: pd.DataFrame) -> None:
    """Integration: the actual TabPFN 3.5 Fast checkpoint (needs `sloth setup` to have run)."""
    res = label.evaluate_holdout(blocks, CFG, holdout_days=7)
    acc = {r["method"]: r["accuracy"] for r in res["rows"]}
    assert acc["TabPFN"] > acc["majority class"]


# --- tagging ----------------------------------------------------------------------------------


def test_tagging_writes_only_block_and_category(tmp_path: Path, fast_model) -> None:
    synthetic.write(tmp_path, days=14, end=dt.date(2026, 10, 18), seed=4)
    answers = iter(["1", "s", "", "My New Category", "q"])
    from rich.console import Console

    console = Console(record=True, width=120)
    n = tagging.run(
        tmp_path, CFG, tmp_path / "labels.csv", limit=10, starter_rules=False, ask=lambda _: next(answers),
        console=console,
    )  # fmt: skip
    assert n == 2
    saved = pd.read_csv(tmp_path / "labels.csv")
    assert list(saved.columns) == label.HAND_LABEL_COLUMNS and len(saved) == 2
    assert "My New Category" in set(saved["category"])
    assert "guess" not in console.export_text().lower()  # the model's answer is never shown


# --- CLI & report -------------------------------------------------------------------------------


def test_cli_hand_eval_with_no_labels(tmp_path: Path) -> None:
    from sloth.cli import app

    synthetic.write(tmp_path, days=7, end=dt.date(2026, 10, 18))
    r = CliRunner().invoke(app, ["eval", "label", "--data", str(tmp_path), "--handlabeled", "labels.csv"])
    assert r.exit_code == 0, r.output
    assert "no hand labels yet" in r.output


def test_report_sections_replace_in_place(tmp_path: Path) -> None:
    p = tmp_path / "eval.md"
    report.update_section("a", "first", p)
    report.update_section("b", "other", p)
    report.update_section("a", "second", p)
    text = p.read_text(encoding="utf-8")
    assert "second" in text and "first" not in text and "other" in text
    assert text.count("<!-- a -->") == 1
