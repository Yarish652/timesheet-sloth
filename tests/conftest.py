import pytest

from sloth import timing


@pytest.fixture(autouse=True)
def _timings_to_tmp(tmp_path, monkeypatch):
    """Keep test runs out of reports/timings.jsonl, which holds the timings the write-up cites."""
    monkeypatch.setattr(timing, "LOG_PATH", tmp_path / "timings.jsonl")


def _tabpfn_weights_cached() -> bool:
    from tabpfn.model_loading import get_cache_dir

    return any(get_cache_dir().glob("tabpfn-v3.5-fast*"))


def pytest_collection_modifyitems(config, items):
    """`slow` tests run the real TabPFN 3.5 Fast checkpoint. Its weights are licence-gated
    (`sloth setup`), so on machines without them (e.g. CI) those tests are skipped, not failed."""
    if _tabpfn_weights_cached():
        return
    skip = pytest.mark.skip(reason="TabPFN 3.5 Fast weights not cached; run `sloth setup`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
