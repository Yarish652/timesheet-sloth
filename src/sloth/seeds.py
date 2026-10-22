"""Starter rules: sloth's own first pass over time your ActivityWatch rules leave Uncategorized.

Generic patterns for common apps and sites, written without looking at anyone's titles. They
use category names from ActivityWatch's default tree, never override a category your rules
already assigned, and are tracked separately (`label_source`) so reports can tell them apart.
The point is to give TabPFN enough seed labels to learn from; TabPFN labels what's left.
"""

from __future__ import annotations

import pandas as pd

from sloth import export

UNCATEGORIZED = export.UNCAT

STARTER_RULES = [
    {
        "name": ["Work", "Programming"],
        "rule": {
            "type": "regex",
            "ignore_case": True,
            "regex": r"\bCode\.exe|Visual Studio Code|Cursor\.exe|WindowsTerminal|\bcmd\.exe|powershell|"
            r"pycharm|idea64|webstorm|jupyter|github\.com|gitlab\.com|stackoverflow\.com|"
            r"stackexchange\.com|localhost|127\.0\.0\.1|pypi\.org|readthedocs|npmjs\.com|"
            r"developer\.mozilla\.org|docs\.python\.org|huggingface\.co",
        },
    },
    {
        "name": ["Work"],
        "rule": {
            "type": "regex",
            "ignore_case": True,
            "regex": r"docs\.google\.com|notion|obsidian|coursera|udemy|arxiv\.org|\.pdf\b|Acrobat|"
            r"WINWORD|EXCEL|POWERPNT|overleaf|chatgpt\.com|claude\.ai|gemini\.google\.com|"
            r"zoom\.us|Zoom\.exe|meet\.google\.com|ms-teams",
        },
    },
    {
        "name": ["Comms", "IM"],
        "rule": {"type": "regex", "ignore_case": True, "regex": r"WhatsApp|Telegram|Discord|Slack|Signal"},
    },
    {
        "name": ["Comms", "Email"],
        "rule": {"type": "regex", "ignore_case": True, "regex": r"mail\.google\.com|outlook|Thunderbird"},
    },
    {
        "name": ["Media", "Video"],
        "rule": {
            "type": "regex",
            "ignore_case": True,
            "regex": r"youtube\.com|netflix|primevideo|hotstar|twitch\.tv|\bvlc\b|crunchyroll",
        },
    },
    {
        "name": ["Media", "Social Media"],
        "rule": {
            "type": "regex",
            "ignore_case": True,
            "regex": r"reddit\.com|twitter\.com|\bx\.com|instagram|linkedin\.com|facebook\.com",
        },
    },
    {
        "name": ["Media", "Music"],
        "rule": {"type": "regex", "ignore_case": True, "regex": r"spotify|music\.youtube\.com"},
    },
    {
        "name": ["Media", "Games"],
        "rule": {"type": "regex", "ignore_case": True, "regex": r"steam\.exe|epicgames|riotclient"},
    },
]

_COMPILED = export.compile_rules(STARTER_RULES)


def apply(events: pd.DataFrame) -> pd.DataFrame:
    """Return events with `rule_category` (your rules), `category` (after starter rules) and
    `label_source` ("your rules", "starter rules" or "none")."""
    ev = events.copy()
    ev["rule_category"] = ev["category"]
    todo = ev["category"] == UNCATEGORIZED
    seeded = [
        export.categorize(_COMPILED, a, t, u)
        for a, t, u in zip(ev.loc[todo, "app"], ev.loc[todo, "title"], ev.loc[todo, "url"], strict=True)
    ]
    ev.loc[todo, "category"] = seeded
    ev["label_source"] = "your rules"
    ev.loc[todo, "label_source"] = "starter rules"
    ev.loc[ev["category"] == UNCATEGORIZED, "label_source"] = "none"
    return ev
