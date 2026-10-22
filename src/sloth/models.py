"""TabPFN model factory: the 3.5 Fast checkpoint on CPU when the installed package has it."""

from __future__ import annotations

from typing import Any

import numpy as np

from sloth.timing import timed


def _version(name: str) -> Any:
    from tabpfn.constants import ModelVersion

    try:
        return ModelVersion(name)
    except ValueError:
        return ModelVersion.V2_5  # older tabpfn releases


def classifier(version: str = "v3.5-fast", **overrides: Any) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier.create_default_for_version(_version(version), device="cpu", **overrides)


def regressor(version: str = "v3.5-fast", **overrides: Any) -> Any:
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor.create_default_for_version(_version(version), device="cpu", **overrides)


def download_weights(version: str = "v3.5-fast") -> None:
    """Fit both models on a toy problem so the weights are fetched (license prompt on first run)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    with timed("tabpfn.download_classifier", version=version):
        classifier(version).fit(X, (X[:, 0] > 0).astype(int))
    with timed("tabpfn.download_regressor", version=version):
        regressor(version).fit(X, X[:, 0])
