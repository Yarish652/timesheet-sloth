"""The agent's tools, written once: Gemma calls them, and (stretch) an MCP server can serve them.

Every tool returns a small JSON-able dict of aggregates: hours, counts, category names and app
executable names. Never window titles, URLs or domains. `agent.py` additionally passes every
result through `privacy.guard` before it can reach a prompt.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from sloth import conditions, features, forecast, label, seeds, sessions
from sloth.config import Config
from sloth.features import AFK, UNCATEGORIZED


def _h(minutes: float) -> float:
    return round(float(minutes) / 60, 2)


@dataclass
class Context:
    """Loaded once per command; tools read from it."""

    data_dir: Path
    cfg: Config
    starter_rules: bool = True
    _cache: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def events(self) -> pd.DataFrame:
        ev = features.load_events(self.data_dir / "events.csv")
        return seeds.apply(ev) if self.starter_rules else ev.assign(rule_category=ev["category"])

    @cached_property
    def pieces(self) -> pd.DataFrame:
        p = features._split_at_blocks(self.events, self.cfg)
        p["date"] = p["block_start"].dt.date
        return p

    @cached_property
    def blocks(self) -> pd.DataFrame:
        return features.build_blocks(self.events, self.cfg)

    @cached_property
    def sessions(self) -> pd.DataFrame:
        return sessions.build_sessions(self.events, self.cfg)

    @property
    def last_date(self) -> dt.date:
        return self.pieces["date"].max()


def parse_range(ctx: Context, spec: str) -> tuple[dt.date, dt.date]:
    """'today', 'yesterday', 'week' (last 7 days), 'YYYY-MM-DD' or 'YYYY-MM-DD..YYYY-MM-DD'."""
    spec = (spec or "week").strip().lower()
    last = ctx.last_date
    if spec == "today":
        return last, last
    if spec == "yesterday":
        return last - dt.timedelta(days=1), last - dt.timedelta(days=1)
    if spec in ("week", "last 7 days"):
        return last - dt.timedelta(days=6), last
    if ".." in spec:
        a, b = spec.split("..", 1)
        return dt.date.fromisoformat(a.strip()), dt.date.fromisoformat(b.strip())
    d = dt.date.fromisoformat(spec)
    return d, d


def _hours_by(p: pd.DataFrame, col: str, top: int = 8) -> dict[str, float]:
    s = p.groupby(col)["seconds"].sum().sort_values(ascending=False) / 60
    s = s[s >= 1]
    if len(s) > top:
        s = pd.concat([s.head(top - 1), pd.Series({"Other": s.iloc[top - 1 :].sum()})])
    return {str(k): _h(v) for k, v in s.items()}


def _focus_sessions(ctx: Context, lo: dt.date, hi: dt.date) -> pd.DataFrame:
    s = ctx.sessions
    return s[(s["date"] >= lo) & (s["date"] <= hi)]


def day_summary(ctx: Context, date: str) -> dict[str, Any]:
    """What one day looked like."""
    d = parse_range(ctx, date)[0]
    p = ctx.pieces[ctx.pieces["date"] == d]
    b = ctx.blocks[ctx.blocks["date"] == d]
    if p.empty:
        return {"date": str(d), "active_hours": 0, "note": "no activity recorded on this day"}
    s = _focus_sessions(ctx, d, d)
    return {
        "date": str(d),
        "weekday": d.strftime("%A"),
        "active_hours": _h(p["seconds"].sum() / 60),
        "first_activity": f"{p['start'].min():%H:%M}",
        "last_activity": f"{p['end'].max():%H:%M}",
        "hours_by_category_your_rules": _hours_by(p, "rule_category"),
        "hours_by_category_after_sloth": _hours_by(p, "category"),
        "uncategorized_hours": _h(p.loc[p["category"] == UNCATEGORIZED, "seconds"].sum() / 60),
        "focus_hours": _h(b["focus_min"].sum()),
        "sessions": len(s),
        "focused_sessions": int(s["focused"].sum()) if len(s) else 0,
        "longest_session_focus_minutes": round(float(s["focus_min"].max()), 0) if len(s) else 0,
        "top_apps_hours": _hours_by(p, "app", top=5),
    }


def week_summary(ctx: Context, start: str = "") -> dict[str, Any]:
    """A 7-day week: totals by category and per day."""
    lo = parse_range(ctx, start)[0] if start else ctx.last_date - dt.timedelta(days=6)
    hi = lo + dt.timedelta(days=6)
    p = ctx.pieces[(ctx.pieces["date"] >= lo) & (ctx.pieces["date"] <= hi)]
    b = ctx.blocks[(ctx.blocks["date"] >= lo) & (ctx.blocks["date"] <= hi)]
    days = []
    for i in range(7):
        d = lo + dt.timedelta(days=i)
        dp, db = p[p["date"] == d], b[b["date"] == d]
        days.append(
            {
                "date": str(d),
                "weekday": d.strftime("%a"),
                "active_hours": _h(dp["seconds"].sum() / 60),
                "work_hours": _h(db["work_min"].sum()),
                "focus_hours": _h(db["focus_min"].sum()),
            }
        )
    best = max(days, key=lambda r: r["focus_hours"])
    return {
        "week": f"{lo}..{hi}",
        "active_hours": _h(p["seconds"].sum() / 60),
        "work_hours": _h(b["work_min"].sum()),
        "focus_hours": _h(b["focus_min"].sum()),
        "hours_by_category_after_sloth": _hours_by(p, "category"),
        "hours_by_category_your_rules": _hours_by(p, "rule_category"),
        "days": days,
        "best_focus_day": best["date"] if best["focus_hours"] > 0 else None,
        "top_apps_hours": _hours_by(p, "app", top=5),
    }


def label_uncategorized(ctx: Context) -> dict[str, Any]:
    """TabPFN's labels for Uncategorized blocks, as counts and hours per predicted category."""
    hand = label.load_hand_labels(ctx.data_dir / "labels.csv", ctx.cfg)
    pred = label.label_uncategorized(ctx.blocks, ctx.cfg, hand)
    if pred.empty:
        return {"uncategorized_blocks": 0}
    g = pred.groupby("predicted").agg(blocks=("predicted", "size"), minutes=("uncategorized_min", "sum"))
    conf = pred.groupby("predicted")["confidence"].mean()
    return {
        "uncategorized_blocks": len(pred),
        "uncategorized_hours": _h(pred["uncategorized_min"].sum()),
        "predicted": {
            str(c): {
                "blocks": int(r["blocks"]),
                "hours": _h(r["minutes"]),
                "mean_confidence": round(float(conf[c]), 2),
            }
            for c, r in g.sort_values("blocks", ascending=False).iterrows()
        },
        "low_confidence_blocks": int((pred["confidence"] < 0.5).sum()),
    }


def forecast_frame(ctx: Context) -> pd.DataFrame:
    """Per-block forecast for the next day, computed once per command (TabPFN is slow on CPU)."""
    if "forecast" not in ctx._cache:
        ctx._cache["forecast"] = forecast.forecast_day(ctx.blocks, ctx.cfg)
    return ctx._cache["forecast"]


def forecast_tomorrow(ctx: Context) -> dict[str, Any]:
    """TabPFN focus forecast for the next day, best windows, and whether the user has a routine."""
    fc = forecast_frame(ctx)
    out = forecast.plan(fc, ctx.blocks, ctx.cfg)
    out["routine_score"] = round(conditions.routine_score(ctx.blocks), 2)
    out["mode"] = conditions.mode(ctx.blocks)
    if out["mode"] == "design":
        out["note"] = "Block-level forecasts of irregular schedules are weak; treat windows as possibilities."
    else:
        out["note"] = "Your focus recurs at similar times; the windows are worth protecting."
    return out


def top_distractions(ctx: Context, range: str = "week") -> dict[str, Any]:
    """Where non-work time went, and when work got interrupted."""
    lo, hi = parse_range(ctx, range)
    p = ctx.pieces[(ctx.pieces["date"] >= lo) & (ctx.pieces["date"] <= hi)]
    b = ctx.blocks[(ctx.blocks["date"] >= lo) & (ctx.blocks["date"] <= hi)]
    nonwork = p[~p["category"].map(lambda c: features._is_work(c, ctx.cfg)) & ~p["category"].isin([UNCATEGORIZED, AFK])]
    shifts_by_hour = b.groupby("hour")["n_shifts"].sum()
    shifts_by_hour = shifts_by_hour[shifts_by_hour > 0].sort_values(ascending=False).head(3)
    return {
        "range": f"{lo}..{hi}",
        "non_work_hours": _h(nonwork["seconds"].sum() / 60),
        "non_work_hours_by_category": _hours_by(nonwork, "category"),
        "non_work_hours_by_app": _hours_by(nonwork, "app", top=5),
        "shifts_from_work": int(b["n_shifts"].sum()),
        "hours_with_most_shifts": {f"{int(h):02d}:00": int(n) for h, n in shifts_by_hour.items()},
        "uncategorized_hours": _h(p.loc[p["category"] == UNCATEGORIZED, "seconds"].sum() / 60),
    }


def focus_conditions(ctx: Context) -> dict[str, Any]:
    """Which start conditions tend to come before focused sessions, and one change to try."""
    s = ctx.sessions
    if len(s) < conditions.MIN_N:
        return {"sessions": len(s), "note": "not enough sessions yet"}
    wi = conditions.what_if(s, ctx.cfg)
    return {
        "sessions": len(s),
        "focused_sessions": int(s["focused"].sum()),
        "focused_rate": round(float(s["focused"].mean()), 2),
        "focused_means": f">= {ctx.cfg.focus_session_min} focus minutes in one session",
        "observed": [
            {**r, "rate": round(r["rate"], 2), "low": round(r["low"], 2), "high": round(r["high"], 2)}
            for r in conditions.insights(s)[:6]
        ],
        "usual_start": wi["usual"],
        "suggestion": (
            {
                **wi["suggestion"],
                "p_usual": round(wi["suggestion"]["p_usual"], 2),
                "p_suggested": round(wi["suggestion"]["p_suggested"], 2),
            }
            if wi["suggestion"]
            else None
        ),
        "why_no_suggestion": wi["reason"],
        "caveat": "Patterns from your own history, not proof of cause.",
        "mode": conditions.mode(ctx.blocks),
    }


TOOLS: dict[str, tuple[Callable[..., dict[str, Any]], str, dict[str, Any]]] = {
    "day_summary": (
        day_summary,
        "Totals for one day: active hours, hours by category (your rules and after sloth's labelling), "
        "focus hours, sessions, top apps. Use for standups.",
        {"date": {"type": "string", "description": "YYYY-MM-DD, 'today' or 'yesterday'"}},
    ),
    "week_summary": (
        week_summary,
        "Totals for a 7-day week starting at `start` (default: the last 7 days): hours by category, per-day "
        "active/work/focus hours. Use for timesheets.",
        {"start": {"type": "string", "description": "YYYY-MM-DD first day, or empty for the last 7 days"}},
    ),
    "label_uncategorized": (
        label_uncategorized,
        "TabPFN's predicted categories for Uncategorized time, with hours and confidence. Slow (~15 s).",
        {},
    ),
    "forecast_tomorrow": (
        forecast_tomorrow,
        "TabPFN forecast of tomorrow's focus hours, best 90-minute windows and whether it looks like a rest "
        "day. Slow (~40 s).",
        {},
    ),
    "top_distractions": (
        top_distractions,
        "Non-work hours by category and app, and how often work was interrupted, for a date range.",
        {
            "range": {
                "type": "string",
                "description": "'today', 'yesterday', 'week' or 'YYYY-MM-DD..YYYY-MM-DD'",
            }
        },
    ),
    "focus_conditions": (
        focus_conditions,
        "Which ways of starting a session tend to lead to focused sessions, from the user's history, plus "
        "at most one suggested change with the evidence behind it.",
        {},
    ),
}


def schemas() -> list[dict[str, Any]]:
    """Ollama / OpenAI-style function schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props)[:1] if props else [],
                },
            },
        }
        for name, (_, desc, props) in TOOLS.items()
    ]


# Tools that run TabPFN. Commands prefetch them before Gemma loads, so the two models never share RAM.
SLOW_TOOLS = ("label_uncategorized", "forecast_tomorrow", "focus_conditions")


def call(ctx: Context, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in TOOLS:
        return {"error": f"unknown tool {name!r}", "available": list(TOOLS)}
    fn, _, props = TOOLS[name]
    kwargs = {k: v for k, v in (args or {}).items() if k in props}
    key = ("tool", name, tuple(sorted(kwargs.items())))
    if key in ctx._cache:
        return ctx._cache[key]
    try:
        result = fn(ctx, **kwargs)
    except (ValueError, KeyError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if name in SLOW_TOOLS:
        ctx._cache[key] = result
    return result


def prefetch(ctx: Context, names: list[str]) -> None:
    """Run TabPFN-backed tools now (before any LLM call) and free the model memory afterwards."""
    import gc

    for name in names:
        call(ctx, name)
    gc.collect()
