"""`sloth` command line."""

from __future__ import annotations

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sloth import config, netguard

app = typer.Typer(help="Timesheet Sloth: your ActivityWatch history, summarised locally.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate the TabPFN models against baselines.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def _main(ctx: typer.Context) -> None:
    config.load_dotenv()
    # TabPFN warns on every CPU fit that a GPU would be faster; we know, it's the point.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"tabpfn(\..*)?$")
    # Everything except one-time setup runs with non-local networking blocked.
    if ctx.invoked_subcommand != "setup":
        netguard.install()


@app.command()
def setup() -> None:
    """One-time: accept the TabPFN license and download the 3.5 Fast weights (needs internet)."""
    from sloth.models import download_weights

    cfg = config.load()
    console.print(f"Downloading TabPFN [bold]{cfg.tabpfn_version}[/] weights...")
    download_weights(cfg.tabpfn_version)
    console.print("[green]Done.[/] From now on sloth runs offline.")


@app.command()
def synth(
    days: int = typer.Option(42, help="Number of days to generate."),
    seed: int = typer.Option(7, help="Random seed."),
    out: Path = typer.Option(Path("demo-data"), help="Output directory."),
) -> None:
    """Write synthetic events.csv and blocks.csv (same format as the exporter) and summarise them."""
    import pandas as pd

    from sloth import synthetic

    ev = synthetic.write(out, days=days, seed=seed)
    n_blocks = len(pd.read_csv(out / "blocks.csv"))
    console.print(f"Wrote {out / 'events.csv'} ({len(ev):,} events) and {out / 'blocks.csv'} ({n_blocks:,} blocks)")

    hours = ev.groupby("category")["duration_s"].sum().sort_values(ascending=False) / 3600
    t = Table(title="Hours per category")
    t.add_column("category")
    t.add_column("hours", justify="right")
    t.add_column("share", justify="right")
    for cat, h in hours.items():
        t.add_row(str(cat), f"{h:.1f}", f"{h / hours.sum():.1%}")
    console.print(t)
    console.print(f"Uncategorized: [bold]{hours.get(synthetic.UNCATEGORIZED, 0) / hours.sum():.1%}[/] of active time")

    local = pd.to_datetime(ev["start"], utc=True).dt.tz_convert(config.load().timezone)
    per_day = ev.assign(date=local.dt.date, wd=local.dt.day_name()).groupby(["date", "wd"])["duration_s"].sum() / 3600
    by_wd = per_day.groupby(level="wd").mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    w = Table(title="Mean active hours per weekday")
    for d in order:
        w.add_column(d[:3], justify="right")
    w.add_row(*(f"{by_wd.get(d, 0):.1f}" for d in order))
    console.print(w)


def _data_dir(data: Path | None) -> Path:
    d = data or config.load().data_dir
    if not (d / "events.csv").exists():
        console.print(f"[red]No events.csv in {d}.[/] Run `python -m sloth.export` or `sloth synth` first.")
        raise typer.Exit(1)
    return d


def _use_starter(d: Path, flag: bool | None) -> bool:
    """Starter rules default on for real exports, off for synthetic data (different category tree)."""
    from sloth import data as data_mod

    return (not data_mod.is_synthetic(d)) if flag is None else flag


def _focus_desc(cfg: config.Config) -> str:
    cats = ", ".join(cfg.work_categories)
    if cfg.focus_rule == "shifts":
        return f"minutes in {cats} within blocks with <= {cfg.focus_max_shifts} shifts from work to non-work"
    return f"minutes in {cats} within blocks with <= {cfg.focus_max_switches} app switches"


_STARTER_HELP = "Apply sloth's starter rules first (default: on for real data, off for synthetic)."


def _score_table(title: str, rows: list[dict]) -> Table:
    t = Table(title=title)
    for col in ("method", "accuracy", "macro-F1", "n_train", "n_test", "fit+predict s"):
        t.add_column(col, justify="left" if col == "method" else "right")
    for r in rows:
        t.add_row(
            r["method"], f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", str(r["n_train"]), str(r["n_test"]),
            f"{r['seconds']:.2f}",
        )  # fmt: skip
    return t


_MD_COLUMNS = [
    ("method", "method"), ("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("n_train", "n_train"),
    ("n_test", "n_test"), ("seconds", "fit+predict s"),
]  # fmt: skip


@app.command()
def label(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show predictions without writing them."),
    tag: bool = typer.Option(False, "--tag", help="Hand-tag the blocks sloth is least sure about."),
    limit: int = typer.Option(20, help="With --tag: how many blocks to ask about."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Predict categories for Uncategorized blocks with TabPFN."""
    import numpy as np

    from sloth import data as data_mod
    from sloth import label as label_mod
    from sloth import tagging

    cfg = config.load()
    d = _data_dir(data)
    use_starter = _use_starter(d, starter_rules)
    if tag:
        tagging.run(d, cfg, d / "labels.csv", limit=limit, starter_rules=use_starter, console=console)
        return

    blocks = data_mod.load_blocks(d, cfg, starter_rules=use_starter)
    hand = label_mod.load_hand_labels(d / "labels.csv", cfg)
    pred = label_mod.label_uncategorized(blocks, cfg, hand)
    console.print(
        f"{len(pred)} Uncategorized blocks ({pred['uncategorized_min'].sum() / 60:.1f} h), "
        f"starter rules {'on' if use_starter else 'off'}"
    )

    counts = pred.groupby("predicted").agg(
        blocks=("predicted", "size"), hours=("uncategorized_min", lambda m: m.sum() / 60),
        mean_conf=("confidence", "mean"),
    ).sort_values("blocks", ascending=False)  # fmt: skip
    t = Table(title="Predicted categories")
    for col in ("category", "blocks", "hours", "mean confidence"):
        t.add_column(col, justify="left" if col == "category" else "right")
    for cat, r in counts.iterrows():
        t.add_row(str(cat), str(int(r["blocks"])), f"{r['hours']:.1f}", f"{r['mean_conf']:.2f}")
    console.print(t)

    hist, edges = np.histogram(pred["confidence"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    h = Table(title="Confidence histogram")
    h.add_column("confidence")
    h.add_column("blocks", justify="right")
    h.add_column("")
    scale = 40 / max(hist.max(), 1)
    for n, lo, hi in zip(hist, edges[:-1], edges[1:], strict=True):
        h.add_row(f"{lo:.1f}-{min(hi, 1):.1f}", str(n), "#" * round(n * scale))
    console.print(h)

    if dry_run:
        console.print("[dim]Dry run: nothing written.[/]")
        return
    out = d / "labels_predicted.csv"
    pred[["block_start", "predicted", "confidence"]].to_csv(out, index=False)
    console.print(f"Wrote {out} (block start, predicted category, confidence; no titles)")


@eval_app.command("label")
def eval_label(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    handlabeled: Path | None = typer.Option(None, help="Score ONLY on hand-tagged blocks in this CSV."),
    holdout_days: int = typer.Option(14, help="Days at the end held out for testing."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """TabPFN labeller vs majority-class and top_app-lookup baselines."""
    from sloth import data as data_mod
    from sloth import label as label_mod
    from sloth import report

    cfg = config.load()
    d = _data_dir(data)
    synthetic = data_mod.is_synthetic(d)
    use_starter = _use_starter(d, starter_rules)
    name = f"{'synthetic' if synthetic else 'real'} data, starter rules {'on' if use_starter else 'off'}"
    blocks = data_mod.load_blocks(d, cfg, starter_rules=use_starter)
    console.print(f"Data: {d} ({name})")

    if handlabeled is not None:
        path = handlabeled if handlabeled.exists() else d / handlabeled.name
        hand = label_mod.load_hand_labels(path, cfg)
        res = label_mod.evaluate_hand(blocks, cfg, hand)
        if res is None:
            console.print(f"no hand labels yet ({path}). Tag some with `sloth label --tag`.")
            return
        console.print(_score_table(f"Hand-labelled blocks only ({path})", res["rows"]))
        report.update_section(
            f"label-hand:{name}",
            f"## Labeller on hand-tagged blocks ({name})\n\nThe honest metric: scored only on blocks tagged "
            f"by hand, which the model never saw.\n\n{report.table(res['rows'], _MD_COLUMNS)}",
        )
        return

    res = label_mod.evaluate_holdout(blocks, cfg, holdout_days)
    console.print(
        f"Train dates {res['train_dates'][0]} .. {res['train_dates'][1]}, "
        f"test dates {res['test_dates'][0]} .. {res['test_dates'][1]}; {len(res['classes'])} classes"
    )
    console.print(_score_table("Holdout: categorized blocks in the last days", res["rows"]))
    md = [
        f"## Labeller ({name})",
        f"Time-based holdout: train {res['train_dates'][0]}..{res['train_dates'][1]}, test "
        f"{res['test_dates'][0]}..{res['test_dates'][1]}, {len(res['classes'])} classes. Test blocks "
        "are ones the rules categorized, so this measures how well TabPFN relearns the rules.",
        report.table(res["rows"], _MD_COLUMNS),
    ]
    truth = label_mod.evaluate_truth(blocks, cfg)
    if truth:
        console.print(_score_table("Uncategorized blocks vs hidden true category (synthetic only)", truth["rows"]))
        md += [
            "Uncategorized blocks scored against the generator's hidden true category (the case that "
            "matters: time the rules missed).",
            report.table(truth["rows"], _MD_COLUMNS),
        ]

    problems = label_mod.leakage_check(res["columns"])
    lk = Table(title="LEAKAGE CHECK: every feature column")
    lk.add_column("column")
    lk.add_column("what it is")
    toks = [c for c in res["columns"] if c.startswith("tok_")]
    for c in res["columns"]:
        if c in toks:
            continue
        lk.add_row(c, "categorical, integer-encoded" if c in label_mod.CATEGORICAL else "numeric")
    if toks:
        lk.add_row(
            ", ".join(toks),
            f"{len(toks)} columns: share of block time whose title/URL contains one vocabulary token "
            "(a number, no text)",
        )
    console.print(lk)
    if problems:
        verdict = "FAIL: " + "; ".join(problems)
    else:
        verdict = "PASS: no feature is the block's category, a rule output, or a raw title/URL."
    console.print(f"[{'green' if not problems else 'red'}]{verdict}[/]")
    md.append(f"Leakage check over {len(res['columns'])} feature columns: {verdict}")
    path = report.update_section(f"label:{name}", "\n\n".join(md))
    console.print(f"[dim]Saved to {path}[/]")


@eval_app.command("focus-definition")
def eval_focus_definition(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """How much work time counts as focus under each rule and threshold (aggregates only)."""
    from sloth import data as data_mod
    from sloth import features, report

    cfg = config.load()
    d = _data_dir(data)
    use_starter = _use_starter(d, starter_rules)
    kind = "synthetic" if data_mod.is_synthetic(d) else "real"
    name = f"{kind} data, starter rules {'on' if use_starter else 'off'}"
    blocks = data_mod.load_blocks(d, cfg, starter_rules=use_starter)
    active_days = blocks.groupby("date")["active_min"].sum()
    days = active_days[active_days > 0].index
    work_total = blocks["work_min"].sum()

    rows = []
    for rule, limits in (("switches", range(0, 11, 2)), ("shifts", range(0, 7))):
        for k in limits:
            rcfg = config.Config(**{**cfg.__dict__, "focus_rule": rule, f"focus_max_{rule}": k})
            ok = [
                features.passes_focus_rule(sw, sh, rcfg)
                for sw, sh in zip(blocks["n_switches"], blocks["n_shifts"], strict=True)
            ]
            focus = blocks["work_min"].where(ok, 0.0)
            per_day = focus.groupby(blocks["date"]).sum().reindex(days).fillna(0) / 60
            current = rule == cfg.focus_rule and k == getattr(cfg, f"focus_max_{rule}")
            rows.append(
                {
                    "rule": f"{rule} <= {k}" + ("  (current)" if current else ""),
                    "median_h": float(per_day.median()),
                    "mean_h": float(per_day.mean()),
                    "kept": float(focus.sum() / work_total) if work_total else 0.0,
                }
            )
    rows.append(
        {
            "rule": "no rule (all work time)",
            "median_h": float((blocks.groupby("date")["work_min"].sum().reindex(days) / 60).median()),
            "mean_h": float(blocks["work_min"].sum() / 60 / len(days)),
            "kept": 1.0,
        }
    )
    t = Table(title=f"Focus definition sensitivity ({name}, {len(days)} active days)")
    for col in ("rule", "focus h/day (median)", "focus h/day (mean)", "% of work kept"):
        t.add_column(col, justify="left" if col == "rule" else "right")
    for r in rows:
        t.add_row(r["rule"], f"{r['median_h']:.2f}", f"{r['mean_h']:.2f}", f"{r['kept']:.0%}")
    console.print(t)
    md_rows = [{**r, "kept": f"{r['kept']:.0%}"} for r in rows]
    path = report.update_section(
        f"focus-definition:{name}",
        f"## Focus definition sensitivity ({name})\n\n"
        "How much of the time in work categories counts as focus under each rule. `switches` is the "
        "original M1-M3 rule (any app change). `shifts` counts only moves from work to a known "
        "non-work category (RescueTime-style), so editor <-> docs/AI chat is free and Uncategorized "
        f"is neutral. {len(days)} active days.\n\n"
        + report.table(
            md_rows,
            [
                ("rule", "rule"),
                ("median_h", "focus h/day (median)"),
                ("mean_h", "focus h/day (mean)"),
                ("kept", "% of work kept"),
            ],
        ),
    )
    console.print(f"[dim]Saved to {path}[/]")


@eval_app.command("forecast")
def eval_forecast(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    backtest_days: int = typer.Option(14, help="Rolling-origin backtest length in days."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Rolling-origin backtest of the focus forecast vs naive baselines."""
    from sloth import data as data_mod
    from sloth import forecast, report

    cfg = config.load()
    d = _data_dir(data)
    use_starter = _use_starter(d, starter_rules)
    kind = "synthetic" if data_mod.is_synthetic(d) else "real"
    limit = getattr(cfg, f"focus_max_{cfg.focus_rule}")
    name = f"{kind} data, starter rules {'on' if use_starter else 'off'}, focus={cfg.focus_rule}<={limit}"
    blocks = data_mod.load_blocks(d, cfg, starter_rules=use_starter)
    console.print(f"Data: {d} ({name}). Focus = {_focus_desc(cfg)}.")
    console.print(f"Backtesting {backtest_days} days, one TabPFN fit per day (rolling origin)...")
    res = forecast.backtest(blocks, cfg, days=backtest_days)

    t = Table(title="Per test day: train range, focus hours, block MAE in minutes per method")
    cols = ("test day", "trained on", "actual h", "TabPFN h", "TabPFN", "last week", "4-wk avg", "zero")
    for col in cols:
        t.add_column(col, justify="left" if col in cols[:2] else "right", no_wrap=True)
    for r in res["per_day"]:
        t.add_row(
            r["date"], f"{r['train_from'][5:]}..{r['train_to'][5:]}", f"{r['actual_h']:.2f}",
            f"{r['TabPFN pred_h']:.2f}", f"{r['TabPFN mae']:.2f}", f"{r['same as last week mae']:.2f}",
            f"{r['rolling 4-week average mae']:.2f}", f"{r['always zero mae']:.2f}",
        )  # fmt: skip
    console.print(t)

    summary = [
        {
            "method": m,
            "block_mae": res["block_mae_min"][m],
            "daily_mae": res["daily_mae_hours"][m],
            "days_won": sum(r[f"{m} mae"] <= min(r[f"{o} mae"] for o in forecast.METHODS) for r in res["per_day"]),
        }
        for m in forecast.METHODS
    ]
    s = Table(title="Overall")
    for col in ("method", "block MAE (min)", "daily-total MAE (h)", "days best (block MAE)"):
        s.add_column(col, justify="left" if col == "method" else "right")
    for r in summary:
        s.add_row(
            r["method"],
            f"{r['block_mae']:.2f}",
            f"{r['daily_mae']:.2f}",
            f"{r['days_won']}/{len(res['per_day'])}",
        )
    console.print(s)
    console.print(
        f"TabPFN 80% interval coverage: {res['interval_80_coverage']:.1%} of blocks | mean actual focus "
        f"{res['mean_daily_focus_hours']:.2f} h/day | runtime {res['seconds']:.0f} s"
    )

    per_day_md = [
        {
            "date": r["date"],
            "train": f"{r['train_from']}..{r['train_to']}",
            "actual_h": r["actual_h"],
            "tabpfn_h": r["TabPFN pred_h"],
            "tabpfn": r["TabPFN mae"],
            "lastweek": r["same as last week mae"],
            "rolling": r["rolling 4-week average mae"],
            "zero": r["always zero mae"],
        }
        for r in res["per_day"]
    ]
    md = "\n\n".join(
        [
            f"## Focus forecast backtest ({name})",
            f"Rolling origin over the last {len(res['per_day'])} complete days: each test day is predicted "
            f"by a model trained only on blocks before that day's 00:00. Focus = {_focus_desc(cfg)}. "
            f"Mean actual focus {res['mean_daily_focus_hours']:.2f} h/day. Runtime "
            f"{res['seconds']:.0f} s.",
            report.table(
                summary,
                [
                    ("method", "method"),
                    ("block_mae", "block MAE (min)"),
                    ("daily_mae", "daily-total MAE (h)"),
                    ("days_won", "days best"),
                ],
            ),
            f"TabPFN 80% predictive interval (10%-90% quantiles) covered {res['interval_80_coverage']:.1%} of blocks.",
            "<details><summary>Per day</summary>\n\n"
            + report.table(
                per_day_md,
                [
                    ("date", "test day"),
                    ("train", "trained on"),
                    ("actual_h", "actual h"),
                    ("tabpfn_h", "TabPFN h"),
                    ("tabpfn", "TabPFN MAE"),
                    ("lastweek", "last-week MAE"),
                    ("rolling", "4-wk MAE"),
                    ("zero", "zero MAE"),
                ],
            )
            + "\n\n</details>",
        ]
    )
    path = report.update_section(f"forecast:{name}", md)
    console.print(f"[dim]Saved to {path}[/]")


@eval_app.command("conditions")
def eval_conditions(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    holdout_days: int = typer.Option(14, help="Days at the end held out for testing."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Does TabPFN know which session starts lead to focus? Holdout vs base-rate baselines."""
    from sloth import conditions, report, tools

    cfg = config.load()
    d = _data_dir(data)
    use_starter = _use_starter(d, starter_rules)
    ctx = tools.Context(d, cfg, use_starter)
    kind = "synthetic" if "true_category" in ctx.events.columns else "real"
    limit = getattr(cfg, f"focus_max_{cfg.focus_rule}")
    name = f"{kind} data, starter rules {'on' if use_starter else 'off'}, focus={cfg.focus_rule}<={limit}"
    s = ctx.sessions
    console.print(
        f"Data: {d} ({name}). {len(s)} sessions (gap >= {cfg.session_gap_min} min); focused = "
        f">= {cfg.focus_session_min} focus min; {s['focused'].mean():.0%} of sessions focused."
    )
    res = conditions.evaluate(s, cfg, holdout_days)
    console.print(
        f"Train {res['train_dates'][0]}..{res['train_dates'][1]}, test {res['test_dates'][0]}.."
        f"{res['test_dates'][1]} (focused rate train {res['focused_rate_train']:.0%}, "
        f"test {res['focused_rate_test']:.0%})"
    )
    t = Table(title="P(session becomes focused | how it starts): holdout")
    cols = [("method", "method"), ("auc", "ROC-AUC (higher)"), ("brier", "Brier (lower)"),
            ("log_loss", "log-loss (lower)"), ("n_train", "n_train"), ("n_test", "n_test"),
            ("seconds", "fit+predict s")]  # fmt: skip
    for _, h in cols:
        t.add_column(h, justify="left" if h == "method" else "right")
    for r in res["rows"]:
        t.add_row(*(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k, _ in cols))
    console.print(t)

    ins = conditions.insights(s)
    it = Table(title=f"Observed focus rates (only conditions seen >= {conditions.MIN_N} times)")
    for h in ("condition", "sessions", "focused", "rate", "90% interval"):
        it.add_column(h, justify="left" if h == "condition" else "right")
    for r in ins:
        it.add_row(r["condition"], str(r["sessions"]), str(r["focused"]), f"{r['rate']:.0%}",
                   f"{r['low']:.0%}-{r['high']:.0%}")  # fmt: skip
    console.print(it)

    wi = conditions.what_if(s, cfg)
    score = conditions.routine_score(ctx.blocks)
    console.print(f"Routine score {score:.2f} -> {conditions.mode(ctx.blocks)} mode")
    if wi["suggestion"]:
        sg = wi["suggestion"]
        console.print(
            f"Suggestion: {sg['change']} (P focused {sg['p_usual']:.0%} -> {sg['p_suggested']:.0%}, "
            f"seen {sg['sessions_seen']} times)"
        )
    else:
        console.print(f"No suggestion: {wi['reason']}")

    feats = conditions.feature_columns()
    from sloth.sessions import START_CATEGORICAL, START_FEATURES

    bad = [c for c in feats if c not in START_FEATURES + START_CATEGORICAL]
    leak = "PASS: features are start-time context only" if not bad else f"FAIL: {bad}"
    console.print(f"LEAKAGE CHECK: {', '.join(feats)} -> {leak}")

    md = [
        f"## Focus conditions ({name})",
        f"{len(s)} sessions (a gap of {cfg.session_gap_min}+ min starts a new one); a session is focused with "
        f">= {cfg.focus_session_min} focus minutes ({s['focused'].mean():.0%} are). Holdout: train "
        f"{res['train_dates'][0]}..{res['train_dates'][1]}, test {res['test_dates'][0]}..{res['test_dates'][1]}.",
        report.table(res["rows"], cols),
        "Observed rates, conditions seen at least "
        f"{conditions.MIN_N} times:\n\n"
        + report.table(
            [{**r, "rate": f"{r['rate']:.0%}", "ci": f"{r['low']:.0%}-{r['high']:.0%}"} for r in ins],
            [
                ("condition", "condition"),
                ("sessions", "sessions"),
                ("focused", "focused"),
                ("rate", "rate"),
                ("ci", "90% interval"),
            ],
        ),
        f"Routine score {score:.2f} ({conditions.mode(ctx.blocks)} mode). "
        + (
            f"Suggestion: {wi['suggestion']['change']} ({wi['suggestion']['p_usual']:.0%} -> "
            f"{wi['suggestion']['p_suggested']:.0%}, seen {wi['suggestion']['sessions_seen']} times)."
            if wi["suggestion"]
            else f"No suggestion: {wi['reason']}."
        ),
        f"Leakage check: {leak}.",
    ]
    path = report.update_section(f"conditions:{name}", "\n\n".join(md))
    console.print(f"[dim]Saved to {path}[/]")


def _context(data: Path | None, starter_rules: bool | None):
    from sloth import tools

    cfg = config.load()
    d = _data_dir(data)
    return tools.Context(d, cfg, _use_starter(d, starter_rules))


def _run_agent(ctx, task: str, trace: bool, report_name: str, title: str, prefetch: tuple[str, ...] = ()) -> None:
    import json

    from rich.markdown import Markdown
    from rich.panel import Panel

    from sloth import agent

    console.print(f"[bold]{agent.banner(ctx.cfg, ctx.data_dir)}[/]")

    def on_call(call: agent.ToolCall) -> None:
        if trace:
            body = json.dumps(call.result, indent=1, default=str)
            console.print(Panel(body, title=f"tool: {call.name}({json.dumps(call.args)})", expand=False))
        else:
            console.print(f"[dim]  ... {call.name}({', '.join(f'{k}={v}' for k, v in call.args.items())})[/]")

    if prefetch:
        from sloth import tools as tools_mod

        console.print(f"[dim]TabPFN first ({', '.join(prefetch)}), so it never shares RAM with Gemma...[/]")
        tools_mod.prefetch(ctx, list(prefetch))
    console.print("[dim]Gemma is thinking (it's a sloth; a few minutes on CPU)...[/]")
    res = agent.run(task, ctx, on_call=on_call)
    console.print(Panel(Markdown(res.text or "_(no answer)_"), title=title))
    if res.unmatched_numbers:
        console.print(f"[yellow]Number check: not found in any tool result: {', '.join(res.unmatched_numbers)}[/]")
    else:
        console.print("[green]Number check: every number in the text appears in a tool result.[/]")
    console.print(f"[dim]{res.llm_calls} LLM calls, {res.llm_seconds:.0f} s · {len(res.calls)} tool calls[/]")

    out = Path(ctx.cfg.reports_dir) / f"{report_name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    check = (
        "every number appears in a tool result"
        if not res.unmatched_numbers else "NOT found in tool results: " + ", ".join(res.unmatched_numbers)
    )  # fmt: skip
    out.write_text(
        f"# {title}\n\n{res.text}\n\n---\n_Number check: {check}. Tools: "
        f"{', '.join(c.name for c in res.calls) or 'none'}. {res.llm_calls} LLM calls, {res.llm_seconds:.0f} s. "
        f"{agent.banner(ctx.cfg, ctx.data_dir)}_\n",
        encoding="utf-8",
    )
    console.print(f"[dim]Saved to {out}[/]")


@app.command("plan-tomorrow")
def plan_tomorrow(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Print the raw forecast only."),
    trace: bool = typer.Option(False, "--trace", help="Show every tool call and result."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Forecast tomorrow's focus hours and suggest when to do hard work."""
    from sloth import forecast, tools

    ctx = _context(data, starter_rules)
    fc = tools.forecast_frame(ctx)
    tr_from, tr_to = fc.attrs["train_dates"]
    t = Table(title=f"Raw forecast for {fc['date'].iloc[0]} (focus min per block; trained on {tr_from}..{tr_to})")
    for col in ("block", "mean", "10%", "50%", "90%", ""):
        t.add_column(col, justify="left" if col in ("block", "") else "right")
    for r in fc.itertuples():
        if r.q90 < 0.5 and r.pred < 0.5:
            continue  # nights and other blocks with nothing expected
        t.add_row(
            f"{r.block_start:%H:%M}", f"{r.pred:.1f}", f"{r.q10:.1f}", f"{r.q50:.1f}", f"{r.q90:.1f}",
            "#" * round(r.pred),
        )  # fmt: skip
    console.print(t)
    p = forecast.plan(fc, ctx.blocks, ctx.cfg)
    console.print(
        f"Predicted focus: [bold]{p['predicted_focus_hours']:.1f} h[/] "
        f"(rough range {p['range_hours'][0]:.1f}-{p['range_hours'][1]:.1f} h; "
        f"your typical day {p['typical_focus_hours']:.1f} h)"
    )
    for w in p["best_windows"]:
        console.print(f"  best window {w['start']}-{w['end']}: ~{w['expected_focus_min']} focus min expected")
    if p["rest_day"]:
        console.print("  Looks like a low-focus day. Rest without guilt.")
    if no_llm:
        return
    day = fc["date"].iloc[0]
    task = (
        f"Plan my {day:%A %Y-%m-%d}. Call forecast_tomorrow and focus_conditions. Write: 1) how much focus to "
        "expect and whether to rest without guilt, 2) the best window(s) for hard work, 3) at most one "
        "suggested change from focus_conditions with its evidence. Under 120 words."
    )
    _run_agent(ctx, task, trace, f"plan-{day}", f"Plan for {day}", prefetch=("forecast_tomorrow", "focus_conditions"))


@app.command()
def standup(
    date: str | None = typer.Option(None, help="YYYY-MM-DD (default: yesterday)."),
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    trace: bool = typer.Option(False, "--trace", help="Show every tool call and result."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Write a daily standup from your activity."""
    from sloth import tools

    ctx = _context(data, starter_rules)
    day = tools.parse_range(ctx, date or "yesterday")[0]
    task = (
        f"Write my daily standup for {day:%A %Y-%m-%d}. Call day_summary with date {day}. Sections: "
        "**Done** (where the time went, by category, with hours), **Focus** (focus hours and sessions), "
        "**Next** (one line; you may call focus_conditions for one suggestion). Under 120 words."
    )
    _run_agent(ctx, task, trace, f"standup-{day}", f"Standup for {day}", prefetch=("focus_conditions",))


@app.command()
def timesheet(
    week: bool = typer.Option(True, "--week", help="Summarise the last 7 days (the only mode for now)."),
    start: str | None = typer.Option(None, help="First day YYYY-MM-DD (default: last 7 days)."),
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    trace: bool = typer.Option(False, "--trace", help="Show every tool call and result."),
    starter_rules: bool | None = typer.Option(None, "--starter-rules/--no-starter-rules", help=_STARTER_HELP),
) -> None:
    """Write a weekly timesheet."""
    ctx = _context(data, starter_rules)
    span = f"starting {start}" if start else "for the last 7 days"
    task = (
        f"Write my weekly timesheet {span}. Call week_summary"
        + (f" with start {start}" if start else "")
        + " and top_distractions with range 'week'. Give a markdown table with one row per day (date, active "
        "hours, work hours, focus hours), then totals by category, then one sentence on where non-work time "
        "went. Use only tool numbers."
    )
    _run_agent(ctx, task, trace, f"timesheet-{start or ctx.last_date}", "Weekly timesheet")


if __name__ == "__main__":
    app()
