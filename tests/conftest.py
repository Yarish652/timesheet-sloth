import pytest

from sloth import timing


@pytest.fixture(autouse=True)
def _timings_to_tmp(tmp_path, monkeypatch):
    """Keep test runs out of reports/timings.jsonl, which holds the timings the write-up cites."""
    monkeypatch.setattr(timing, "LOG_PATH", tmp_path / "timings.jsonl")
