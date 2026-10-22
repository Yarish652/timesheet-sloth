"""Load a data dir (exporter or synthetic output) into model-ready blocks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sloth import features, seeds
from sloth.config import Config


def load_blocks(data_dir: Path, cfg: Config, starter_rules: bool = True) -> pd.DataFrame:
    events = features.load_events(data_dir / "events.csv")
    if starter_rules:
        events = seeds.apply(events)
    return features.build_blocks(events, cfg)


def is_synthetic(data_dir: Path) -> bool:
    with (data_dir / "events.csv").open(encoding="utf-8") as f:
        return "true_category" in f.readline()
