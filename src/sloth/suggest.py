"""Suggestions instead of predictions: what to open first, and what to resume, for the next sessions.

`build` does the slow part (one TabPFN call over a grid of possible return times) so that `pick`
and `render` can answer instantly when the user sits down. Nothing here needs the LLM.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from sloth import features
from sloth.config import Config
from sloth.sessions import hour_bucket

GRID_HOURS = 24  # one candidate return time per hour, starting at the build time
EDITORS = ("Code.exe", "Cursor.exe", "Code - Insiders.exe")
_EDITOR_SUFFIX = re.compile(r"\s-\s(Visual Studio Code|Cursor)(\s-\s.*)?$")


def _round_hour(t: pd.Timestamp) -> pd.Timestamp:
    """Start of the current hour: the grid must cover 'right now', not only the hours ahead."""
    return t.floor("h")


def option_grid(sessions: pd.DataFrame, cfg: Config, now: pd.Timestamp, min_n: int | None = None) -> list[dict]:
    """For each of the next GRID_HOURS return times, score every way of starting the user has
    actually used in that part of the day (n >= min_n), in a single TabPFN call."""
    from sloth import conditions  # lazy: scikit-learn costs ~1.7 s to import, and `sloth next` never needs it

    min_n = conditions.MIN_N if min_n is None else min_n
    combos = sessions.groupby(["hour_bucket", "starter_family"]).size()
    supported = combos[combos >= min_n]
    if supported.empty:
        return []
    times = [_round_hour(now) + pd.Timedelta(hours=i) for i in range(GRID_HOURS)]
    rows, keys = [], []
    for t in times:
        bucket = hour_bucket(t.hour)
        for fam in supported.loc[bucket].index if bucket in supported.index.get_level_values(0) else []:
            rows.append(conditions.start_row(sessions, t, fam))
            keys.append((t, bucket, fam))
    if not rows:
        return []
    p = conditions.predict_proba(sessions, pd.DataFrame(rows), cfg)

    grid: list[dict] = []
    for t in times:
        bucket = hour_bucket(t.hour)
        opts = [
            {"starter_family": f, "p_focused": float(pi), "sessions_seen": int(supported[(b, f)])}
            for (tt, b, f), pi in zip(keys, p, strict=True)
            if tt == t
        ]
        if not opts:
            continue
        in_bucket = combos.loc[bucket]
        usual_fam = in_bucket.idxmax()
        usual = next((o for o in opts if o["starter_family"] == usual_fam), None)
        for o in opts:
            o["feasibility"] = float(in_bucket[o["starter_family"]] / in_bucket.sum())
            o["gain"] = o["p_focused"] - (usual["p_focused"] if usual else o["p_focused"])
            o["score"] = o["gain"] * o["feasibility"]
        ranked = sorted(opts, key=lambda o: -o["score"])
        best = ranked[0] if ranked[0]["gain"] >= conditions.MIN_GAIN else None
        grid.append(
            {
                "at": t.isoformat(),
                "bucket": bucket,
                "usual_family": usual_fam,
                "usual_p": usual["p_focused"] if usual else None,
                "best": best,
                "options": ranked,
            }
        )
    return grid


def resume_target(pieces: pd.DataFrame, cfg: Config, with_project: bool = False) -> dict[str, Any] | None:
    """The most recent stretch of work: its top category and app, and (opt-in, local only) project."""
    work = pieces[pieces["category"].map(lambda c: features._is_work(c, cfg))]
    if work.empty:
        return None
    last_end = work["end"].max()
    recent = work[work["end"] >= last_end - pd.Timedelta(hours=3)]
    out: dict[str, Any] = {
        "category": recent.groupby("category")["seconds"].sum().idxmax(),
        "app": recent.groupby("app")["seconds"].sum().idxmax(),
        "minutes": round(float(recent["seconds"].sum() / 60)),
        "ended": last_end.isoformat(),
    }
    if with_project:
        out["project"] = editor_project(recent)
    return out


def editor_project(pieces: pd.DataFrame) -> str | None:
    """Workspace name from editor titles like 'main.py - my-project - Visual Studio Code'."""
    ed = pieces[pieces["app"].isin(EDITORS)]
    names: dict[str, float] = {}
    for title, secs in zip(ed["title"], ed["seconds"], strict=True):
        stem = _EDITOR_SUFFIX.sub("", str(title))
        parts = [x.strip() for x in stem.split(" - ") if x.strip()]
        if len(parts) >= 2:
            names[parts[-1]] = names.get(parts[-1], 0.0) + float(secs)
    return max(names, key=names.get) if names else None


def build(ctx: Any, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Everything `sloth next` needs, computed ahead of time. `ctx` is a `tools.Context`."""
    from sloth import conditions

    cfg = ctx.cfg
    now = now or pd.Timestamp.now(tz=cfg.timezone)
    s = ctx.sessions
    payload: dict[str, Any] = {
        "built_at": now.isoformat(),
        "sessions": len(s),
        "focused_rate": round(float(s["focused"].mean()), 3) if len(s) else None,
        "grid": option_grid(s, cfg, now) if len(s) >= conditions.MIN_N else [],
        "resume": None,
        "local_only": {},
    }
    target = resume_target(ctx.pieces, cfg, with_project=cfg.suggest_projects)
    if target:
        project = target.pop("project", None)
        payload["resume"] = target
        if project:
            payload["local_only"]["project"] = project  # shown to the user, never to an LLM
    return payload


def pick(payload: dict[str, Any], now: pd.Timestamp, tolerance_min: int = 60) -> dict[str, Any] | None:
    """The grid entry nearest to `now` (within tolerance), or None."""
    best, best_gap = None, None
    for entry in payload.get("grid", []):
        gap = abs((pd.Timestamp(entry["at"]) - now).total_seconds()) / 60
        if gap <= tolerance_min and (best_gap is None or gap < best_gap):
            best, best_gap = entry, gap
    return best


def _ago(ended: str, now: pd.Timestamp) -> str:
    h = (now - pd.Timestamp(ended)).total_seconds() / 3600
    return f"{h * 60:.0f} min ago" if h < 1 else f"{h:.0f} h ago" if h < 48 else f"{h / 24:.0f} days ago"


def facts(payload: dict[str, Any], entry: dict[str, Any] | None, now: pd.Timestamp) -> dict[str, Any]:
    """The numbers a suggestion is made of: what the template and (optionally) Gemma may use."""
    out: dict[str, Any] = {"part_of_day": entry["bucket"] if entry else None}
    if entry and entry["best"]:
        b = entry["best"]
        out |= {
            "open_first": b["starter_family"],
            "p_focused_if_so": round(b["p_focused"], 2),
            "usual_first": entry["usual_family"],
            "p_focused_usual": round(entry["usual_p"], 2) if entry["usual_p"] is not None else None,
            "sessions_seen": b["sessions_seen"],
        }
    r = payload.get("resume")
    if r:
        out |= {"resume_category": r["category"], "resume_app": r["app"], "resume_ended": _ago(r["ended"], now)}
    return out


def render(payload: dict[str, Any], entry: dict[str, Any] | None, now: pd.Timestamp, local: bool = True) -> str:
    """Plain-language suggestion. `local=True` may include the project name (never sent anywhere)."""
    f = facts(payload, entry, now)
    lines = []
    polished = payload.get("polished", {}).get(f["part_of_day"] or "")
    if polished and "open_first" in f:
        lines.append(polished)
    elif "open_first" in f:
        lines.append(
            f"{f['part_of_day'].split(' ')[0].capitalize()} sessions that start with {f['open_first']} tend to "
            f"become focused {f['p_focused_if_so']:.0%} of the time (from {f['sessions_seen']} sessions), vs "
            f"{f['p_focused_usual']:.0%} for your usual {f['usual_first']} start."
        )
    if "resume_category" in f:
        where = f"{f['resume_category']} in {f['resume_app']}"
        project = payload.get("local_only", {}).get("project") if local else None
        lines.append(f"Pick up where you left off: {where}" + (f" ({project})" if project else "") +
                     f", last touched {f['resume_ended']}.")  # fmt: skip
    if not lines:
        lines.append("Nothing to suggest yet: sloth needs more history. Rest without guilt.")
    return " ".join(lines)


POLISH_KEYS = ("part_of_day", "open_first", "p_focused_if_so", "usual_first", "p_focused_usual", "sessions_seen")


def polish(f: dict[str, Any], ctx: Any) -> str | None:
    """Let Gemma reword the "open first" suggestion in one sentence. It gets only category names and
    numbers (checked by the privacy guard); output with any number not in `f` is discarded."""
    import httpx

    from sloth import agent
    from sloth.privacy import Guard

    facts_only = {k: f[k] for k in POLISH_KEYS if k in f}
    if "open_first" not in facts_only:
        return None
    prompt = (
        "Rewrite this as one friendly, slightly lazy sentence of at most 40 words for the user. Use only these "
        "facts, keep the numbers (as percentages), say 'tends to', and mention the number of sessions: "
        + Guard(ctx.events).check(facts_only)
    )
    messages = [{"role": "system", "content": agent.SYSTEM}, {"role": "user", "content": prompt}]
    try:
        with httpx.Client() as client:
            data = agent._chat(client, ctx.cfg, messages, use_tools=False)
    except (httpx.HTTPError, OSError):
        return None
    text = (data.get("message", {}).get("content") or "").strip()
    # Keep Gemma's wording only if every number checks out AND it still says what to open first.
    if not text or agent.check_numbers(text, [facts_only]) or facts_only["open_first"].lower() not in text.lower():
        return None
    return text
