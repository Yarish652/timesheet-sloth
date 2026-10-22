"""Synthetic ActivityWatch-style data for development, tests and the demo video.

Persona: a student developer in IST. Weekdays lean towards programming, study and meetings;
evenings and weekends lean towards video, games and chat. About 30% of active time is left
Uncategorized because the (fake) category rules don't cover every app or site, which is the
situation the labeller is meant to fix.

Every event also carries a `true_category`, which the real exporter can't know. It lets us
score the labeller on genuinely-uncategorized time, not just on a holdout of categorized time.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UNCATEGORIZED = "Uncategorized"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EVENT_COLUMNS = ["start", "end", "duration_s", "app", "title", "url", "category"]  # sloth.export


@dataclass(frozen=True)
class Activity:
    """One kind of thing the persona does: an app (or site) with its title generator."""

    app: str
    true_category: str
    titles: tuple[str, ...]
    url: str = ""  # domain for browser activities, empty for desktop apps
    covered: bool = True  # do the user's category rules match it?
    mean_s: float = 120.0  # mean event duration in seconds


BROWSER = "chrome.exe"
FILES = ("main.py", "features.py", "README.md", "train.ipynb", "app.tsx", "utils.ts", "api.rs")
PROJECTS = ("timesheet-sloth", "microgpt-fpga", "dsa-notes", "portfolio")
TOPICS = ("transformers", "graph algorithms", "rust lifetimes", "react hooks", "fpga timing")

ACTIVITIES: dict[str, Activity] = {
    a.app + a.url: a
    for a in [
        Activity(
            "Code.exe",
            "Work > Programming",
            ("{file} - {project} - Visual Studio Code",),
            mean_s=240,
        ),
        Activity(
            "WindowsTerminal.exe",
            "Work > Programming",
            ("pytest", "uv run {file}", "git log", "npm run dev"),
            mean_s=60,
        ),
        Activity(
            "pycharm64.exe",
            "Work > Programming",
            ("{project} - {file}",),
            covered=False,
            mean_s=240,
        ),
        Activity(
            BROWSER,
            "Work > Programming",
            ("{project}: Pull request #{n} - GitHub",),
            url="github.com",
            mean_s=90,
        ),
        Activity(
            BROWSER,
            "Work > Programming",
            ("python - How to fix {topic} error - Stack Overflow",),
            url="stackoverflow.com",
            covered=False,
            mean_s=70,
        ),
        Activity(
            BROWSER,
            "Work > Programming",
            ("localhost:{port}", "React App"),
            url="localhost",
            covered=False,
            mean_s=50,
        ),
        Activity(
            BROWSER,
            "Work > Programming",
            ("{topic} - docs",),
            url="docs.python.org",
            covered=False,
            mean_s=80,
        ),
        Activity(
            "Zoom.exe",
            "Work > Meetings",
            ("Zoom Meeting", "Zoom Workplace"),
            covered=False,
            mean_s=900,
        ),
        Activity(
            "ms-teams.exe",
            "Work > Meetings",
            ("Standup | Microsoft Teams", "Chat | Microsoft Teams"),
            mean_s=600,
        ),
        Activity("Obsidian.exe", "Work > Study", ("{topic} - vault - Obsidian",), mean_s=300),
        Activity("Acrobat.exe", "Work > Study", ("lecture_{n}.pdf - Adobe Acrobat",), mean_s=360),
        Activity(BROWSER, "Work > Study", ("{topic} | Coursera",), url="coursera.org", mean_s=400),
        Activity(
            BROWSER,
            "Work > Study",
            ("[{n}.0{port}] {topic} - arXiv",),
            url="arxiv.org",
            covered=False,
            mean_s=300,
        ),
        Activity(
            "Notion.exe",
            "Work > Study",
            ("{topic} notes", "Semester plan"),
            covered=False,
            mean_s=200,
        ),
        Activity(
            "Discord.exe",
            "Comms > Chat",
            ("#general | dev-friends - Discord", "@friend - Discord"),
            mean_s=45,
        ),
        Activity("WhatsApp.exe", "Comms > Chat", ("WhatsApp",), covered=False, mean_s=40),
        Activity(BROWSER, "Comms > Chat", ("Inbox ({n}) - Gmail",), url="mail.google.com", mean_s=60),
        Activity(
            BROWSER,
            "Media > Video",
            ("{topic} explained - YouTube", "lofi beats - YouTube"),
            url="youtube.com",
            mean_s=500,
        ),
        Activity(BROWSER, "Media > Video", ("Netflix",), url="netflix.com", mean_s=1200),
        Activity("vlc.exe", "Media > Video", ("episode_{n}.mkv - VLC media player",), covered=False, mean_s=1200),
        Activity(BROWSER, "Media > Social", ("r/programming - Reddit",), url="reddit.com", mean_s=90),
        Activity(BROWSER, "Media > Social", ("Home / X",), url="x.com", covered=False, mean_s=70),
        Activity("steam.exe", "Media > Games", ("Steam",), mean_s=120),
        Activity("HollowKnight.exe", "Media > Games", ("Hollow Knight",), mean_s=1500),
        Activity(
            "explorer.exe",
            "Uncategorized",
            ("Downloads", "File Explorer"),
            covered=False,
            mean_s=20,
        ),
    ]
}

# Each mode is what a 30-minute block is "about": activity keys with weights.
MODES: dict[str, dict[str, float]] = {
    "programming": {
        "Code.exe": 5,
        "WindowsTerminal.exe": 2,
        "pycharm64.exe": 1.2,
        "chrome.exegithub.com": 1,
        "chrome.exestackoverflow.com": 1.2,
        "chrome.exelocalhost": 0.8,
        "chrome.exedocs.python.org": 0.8,
    },
    "meeting": {"Zoom.exe": 3, "ms-teams.exe": 2},
    "study": {
        "Obsidian.exe": 2,
        "Acrobat.exe": 2,
        "chrome.execoursera.org": 2,
        "chrome.exearxiv.org": 1.5,
        "Notion.exe": 1.5,
    },
    "chat": {"Discord.exe": 3, "WhatsApp.exe": 2.5, "chrome.exemail.google.com": 1},
    "video": {"chrome.exeyoutube.com": 3, "chrome.exenetflix.com": 1.5, "vlc.exe": 1.5},
    "social": {"chrome.exereddit.com": 2, "chrome.exex.com": 2},
    "games": {"steam.exe": 0.5, "HollowKnight.exe": 3},
}
WORK_MODES = ("programming", "meeting", "study")
DISTRACTIONS = {"chat": 0.6, "social": 0.4}  # what interrupts a focused block


def _mode_weights(hour: int, weekend: bool) -> dict[str, float]:
    if weekend:
        if hour < 12:
            return {"video": 3, "chat": 2, "social": 2, "programming": 1}
        if hour < 19:
            return {"programming": 2, "games": 3, "video": 2, "chat": 1, "study": 0.5}
        return {"games": 3, "video": 3, "chat": 2, "social": 1}
    if hour < 12:
        return {"programming": 3, "study": 3, "meeting": 1, "chat": 1, "social": 0.5}
    if hour < 18:
        return {
            "programming": 4,
            "meeting": 1.5,
            "study": 1.5,
            "chat": 1,
            "social": 1,
            "video": 0.5,
        }
    if hour < 22:
        return {"video": 3, "chat": 2, "games": 2, "programming": 1.5, "social": 1}
    return {"programming": 2, "video": 2, "social": 1.5, "chat": 1}


# How much of the waking day is spent at the laptop, by weekday (Mon=0).
DAY_FACTOR = {0: 1.0, 1: 0.95, 2: 0.85, 3: 1.0, 4: 0.9, 5: 0.6, 6: 0.75}


def _active_probability(hour: float, wake: float, sleep: float, weekday: int = 0) -> float:
    """Probability the laptop is in use during a block starting at `hour` (0-24, sleep may be >24)."""
    h = hour if hour >= 6 else hour + 24
    if h < wake or h >= sleep:
        return 0.0
    if weekday == 2 and 14 <= h < 17:  # Wednesday lab, away from the laptop
        return 0.1
    if weekday == 4 and h >= 19:  # Friday evening out
        return 0.2
    if 13 <= h < 14:  # lunch
        return 0.3
    if 19.5 <= h < 20.5:  # dinner
        return 0.35
    return 0.85 * DAY_FACTOR[weekday]


def _fill(template: str, rng: np.random.Generator) -> str:
    return template.format(
        file=rng.choice(FILES),
        project=rng.choice(PROJECTS),
        topic=rng.choice(TOPICS),
        n=int(rng.integers(1, 400)),
        port=int(rng.choice([3000, 5173, 8000, 8080])),
    )


def generate(
    days: int = 42,
    end: dt.date | None = None,
    seed: int = 7,
    tz: dt.tzinfo = IST,
) -> pd.DataFrame:
    """Generate `days` of AFK-filtered window/web events ending on `end` (default: yesterday).

    Returns a frame with the exporter's event columns (tz-aware `start`/`end`) plus `true_category`.
    """
    rng = np.random.default_rng(seed)
    end = end or dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days - 1)
    rows: list[dict[str, object]] = []

    for d in range(days):
        day = start + dt.timedelta(days=d)
        weekend = day.weekday() >= 5
        # Deadline weeks are more focused; this gives the forecaster week-to-week signal.
        week_focus = 1.0 + 0.5 * np.sin(2 * np.pi * d / 21) + rng.normal(0, 0.1)
        wake = rng.normal(10.0 if weekend else 8.5, 0.6)
        sleep = rng.normal(25.5 if weekend else 24.8, 0.5)
        mode: str | None = None

        for block in range(48):
            hour = block / 2
            if rng.random() >= _active_probability(hour, wake, sleep, day.weekday()):
                mode = None
                continue
            weights = _mode_weights(int(hour), weekend)
            for m in WORK_MODES:
                if m in weights:
                    weights[m] *= max(week_focus, 0.2)
            if mode is not None and rng.random() < 0.55:
                pass  # persistence: keep doing what we were doing
            else:
                names = list(weights)
                p = np.array([weights[n] for n in names])
                mode = str(rng.choice(names, p=p / p.sum()))

            block_start = dt.datetime.combine(day, dt.time(), tzinfo=tz) + dt.timedelta(minutes=30 * block)
            t = float(rng.uniform(0, 300))
            active_until = 1800 * float(rng.uniform(0.6, 1.0))
            # Some focused blocks are fragmented (frequent distractions), some are deep work.
            distract_p = float(rng.choice([0.05, 0.15, 0.35])) if mode in WORK_MODES else 0.1
            while t < active_until:
                pool = MODES[mode]
                if rng.random() < distract_p and mode not in DISTRACTIONS:
                    dmode = str(rng.choice(list(DISTRACTIONS), p=list(DISTRACTIONS.values())))
                    pool = MODES[dmode]
                    if rng.random() < 0.15:
                        pool = {"explorer.exe": 1.0}
                keys = list(pool)
                w = np.array([pool[k] for k in keys])
                act = ACTIVITIES[str(rng.choice(keys, p=w / w.sum()))]
                dur = round(float(min(rng.exponential(act.mean_s) + 5, active_until - t)), 1)
                ts = block_start + dt.timedelta(seconds=round(t, 3))
                rows.append(
                    {
                        "start": ts,
                        "end": ts + dt.timedelta(seconds=dur),
                        "duration_s": dur,
                        "app": act.app,
                        "title": _fill(str(rng.choice(act.titles)), rng),
                        "url": f"https://{act.url}/" if act.url else "",
                        "category": act.true_category if act.covered else UNCATEGORIZED,
                        "true_category": act.true_category,
                    }
                )
                t += dur + float(rng.exponential(10))

    return pd.DataFrame(rows, columns=[*EVENT_COLUMNS, "true_category"])


def write(out_dir: Path, block_minutes: int = 30, **kwargs: Any) -> pd.DataFrame:
    """Write synthetic `events.csv` and `blocks.csv` into `out_dir`, exactly as the exporter would.

    `events.csv` gets an extra `true_category` column. Returns the events frame.
    """
    from sloth import export

    out_dir.mkdir(parents=True, exist_ok=True)
    events = generate(**kwargs)
    rows = events.to_dict("records")
    export.write_events(rows, out_dir / "events.csv", extra_fields=["true_category"])
    export.write_blocks(export.aggregate_blocks(rows, block_minutes), out_dir / "blocks.csv")
    return events
