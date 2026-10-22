# Timesheet Sloth

Fully local agent over ActivityWatch: standups, timesheets, TabPFN auto-labelling and focus
forecasts, Gemma 4 (Ollama) writing the prose. Built for DEV Hacktoberfest Open-Source AI
Challenge, Week 3. Deadline Sun 25 Oct 2026, 23:59 PDT. Replaced the earlier aw-mcp GSoC
scaffold on 2026-10-22 (old plan backed up outside the repo).

## Privacy rule (for coding assistants)

- NEVER open, cat, head, grep or print `data/events.csv`, `data/blocks.csv`, or any real export.
  They contain window titles and URLs.
- Develop and test against `sloth.synthetic`. On real data, run scripts that print aggregates only
  (counts, hours per category, metrics).

## Working agreement

- Claude writes the code; commit after each milestone (M1–M5). No `Co-Authored-By: Claude` trailer.
- Log decisions, surprises and breakage in `NOTES.md`.
- Only localhost network calls at runtime: ActivityWatch :5600, Ollama :11434.

## Commands

- `uv sync` · `uv run pytest` · `uv run ruff check` · `uv run ruff format`
- `uv run sloth --help`
