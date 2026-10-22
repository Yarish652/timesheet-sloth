"""`sloth` command line."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sloth import config, netguard

app = typer.Typer(
    help="Timesheet Sloth: your ActivityWatch history, summarised locally.", no_args_is_help=True
)
eval_app = typer.Typer(help="Evaluate the TabPFN models against baselines.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


def _todo(milestone: str) -> None:
    console.print(f"[yellow]Not implemented yet[/] (lands in {milestone}).")
    raise typer.Exit(2)


@app.callback()
def _main(ctx: typer.Context) -> None:
    config.load_dotenv()
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
    console.print(
        f"Wrote {out / 'events.csv'} ({len(ev):,} events) and {out / 'blocks.csv'} ({n_blocks:,} blocks)"
    )

    hours = ev.groupby("category")["duration_s"].sum().sort_values(ascending=False) / 3600
    t = Table(title="Hours per category")
    t.add_column("category")
    t.add_column("hours", justify="right")
    t.add_column("share", justify="right")
    for cat, h in hours.items():
        t.add_row(str(cat), f"{h:.1f}", f"{h / hours.sum():.1%}")
    console.print(t)
    console.print(
        f"Uncategorized: [bold]{hours.get(synthetic.UNCATEGORIZED, 0) / hours.sum():.1%}[/] of active time"
    )

    local = pd.to_datetime(ev["start"], utc=True).dt.tz_convert(config.load().timezone)
    per_day = (
        ev.assign(date=local.dt.date, wd=local.dt.day_name()).groupby(["date", "wd"])["duration_s"].sum()
        / 3600
    )
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
        console.print(
            _score_table("Uncategorized blocks vs hidden true category (synthetic only)", truth["rows"])
        )
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


@eval_app.command("forecast")
def eval_forecast(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    backtest_days: int = typer.Option(14, help="Rolling-origin backtest length in days."),
) -> None:
    """Rolling-origin backtest of the focus forecast vs naive baselines."""
    _todo("M3")


@app.command("plan-tomorrow")
def plan_tomorrow(
    no_llm: bool = typer.Option(False, "--no-llm", help="Print the raw forecast only."),
) -> None:
    """Forecast tomorrow's focus hours and suggest when to do hard work."""
    _todo("M3/M4")


@app.command()
def standup(
    date: str | None = typer.Option(None, help="YYYY-MM-DD (default: yesterday)."),
    trace: bool = typer.Option(False, "--trace", help="Show every tool call and result."),
) -> None:
    """Write a daily standup from your activity."""
    _todo("M4")


@app.command()
def timesheet(
    week: bool = typer.Option(False, "--week", help="Summarise the last full week."),
    trace: bool = typer.Option(False, "--trace", help="Show every tool call and result."),
) -> None:
    """Write a weekly timesheet."""
    _todo("M4")


if __name__ == "__main__":
    app()
