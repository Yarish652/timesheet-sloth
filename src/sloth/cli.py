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


@app.command()
def label(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show predictions without writing them."),
) -> None:
    """Predict categories for Uncategorized blocks with TabPFN."""
    _todo("M2")


@eval_app.command("label")
def eval_label(
    data: Path | None = typer.Option(None, help="Data dir (default: $SLOTH_DATA_DIR)."),
    handlabeled: Path | None = typer.Option(None, help="Score only on hand-tagged blocks in this CSV."),
) -> None:
    """TabPFN labeller vs majority-class and top_app-lookup baselines."""
    _todo("M2")


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
