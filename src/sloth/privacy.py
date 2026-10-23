"""Fail-closed check that nothing headed for the LLM contains a window title or URL.

Tools are written to return aggregates only; this is the second, independent line. It scans
the serialised payload for every title and URL in the user's data and raises instead of
letting one through.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import pandas as pd

MIN_LEN = 6  # shorter strings ("Inbox", "Home") are too common to be identifying
MIN_WORD_LEN = 12  # a single plain word ("History", "Downloads") isn't identifying unless long


def _identifying(s: str) -> bool:
    return len(s) >= MIN_LEN and (not s.isalpha() or len(s) >= MIN_WORD_LEN)


class PrivacyError(RuntimeError):
    pass


class Guard:
    def __init__(self, events: pd.DataFrame):
        allowed = {str(a) for a in events["app"].unique()} | {str(c) for c in events["category"].unique()}
        if "rule_category" in events.columns:
            allowed |= {str(c) for c in events["rule_category"].unique()}
        allowed_text = "\n".join(allowed).lower()
        secrets: set[str] = set()
        for title in events["title"].dropna().unique():
            t = str(title).strip().lower()
            if _identifying(t) and t not in allowed_text:  # an app/category name is not a secret
                secrets.add(t)
        for url in events["url"].dropna().unique():
            u = str(url).strip().lower()
            if len(u) >= MIN_LEN:
                secrets.add(u)
                host = (urlparse(u).hostname or "").removeprefix("www.")
                if len(host) >= MIN_LEN and host not in allowed_text:
                    secrets.add(host)
        self.secrets = secrets

    def check(self, payload: Any) -> str:
        """Return the JSON text for `payload`, or raise PrivacyError if it contains a secret."""
        text = payload if isinstance(payload, str) else json.dumps(payload, default=str, ensure_ascii=False)
        low = text.lower()
        hits = sum(1 for s in self.secrets if s in low)
        if hits:
            # Never echo the matched text: the error itself must not leak it.
            raise PrivacyError(f"blocked: payload contains {hits} window title(s)/URL(s) from your data")
        return text
