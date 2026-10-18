# aw-mcp — ActivityWatch MCP Server

Privacy-first MCP server that lets an AI answer questions about the user's time from their
ActivityWatch data, sharing **summaries only** (never window titles or URLs) — and proving it.
Source of truth: `C:\Users\kyari\Downloads\ActivityWatch_MCP_Server_Project_Plan_v2.pdf` (19 Oct 2026).

## How to work with me (read first)

- This is a **learning project** for GSoC 2027. Work **one phase, one checklist item at a time**.
  Explain the why, point to files/docs, and let me write the code unless I ask you to write it.
- Before starting an item, say which phase/item it is. Don't jump ahead to later phases.
- Never commit, push, or publish unless I ask. No `Co-Authored-By: Claude` trailer on commits.
- When a phase's "Done when" is met, tick it in the Progress tracker below and add the date.

## Environment

- Package manager: **uv** (`C:\Users\kyari\AppData\Roaming\Python\Python314\Scripts\uv.exe`,
  or `py -m uv` if not on PATH). Python 3.13 venv in `.venv`; support 3.11–3.13.
- Run: `uv run pytest` · `uv run ruff check` · `uv run ruff format` · `uv run pyright` (strict)
- Inspector: `uv run mcp dev src/aw_mcp/server.py`
- aw-server at `localhost:5600`; testing instances on port `5666`.
- Local clone of aw-client (the library we wrap): `C:\Users\kyari\aw-client`.

## Things old tutorials get wrong

- MCP spec **2026-07-28** is **stateless** (no handshake, no sessions).
- Python SDK **v2**: `FastMCP` is now **`MCPServer`** (`from mcp.server import MCPServer`).
  Ignore tutorials using the old name.
- ActivityWatch **v0.14**: categories live on the server (Settings API), optional API keys,
  title filter, synced buckets (`-synced-from-`), canonical query aimed at AI use.

## Rules of the house

1. **Pure core, thin edges** — `core/` has no I/O; adapters do the talking (hexagonal design).
2. **No field = no leak** — default output models have no title/URL fields at all.
3. **Fail closed** — egress guard returns an error rather than leak.
4. **The past is cached** — finished days are computed once.
5. **stdout is the wire** — never `print()`; logs go to stderr (ruff `T20` enforces this).
6. **Small, honest outputs** — top-N + "Other" so parts always add up to the total; < 1 KB.

All tools are read-only (`read_only_hint=True, open_world_hint=False`) with Pydantic return
types (structured output). Tool descriptions are written for a model, not a human. Write each
tool's design (description, input, output, errors, never-returns, budget) before coding it.

## Repository layout (target)

```
src/aw_mcp/
  server.py        # MCP surface: tools, resources, prompts (thin)
  middleware/      # egress guard, audit receipts
  core/            # pure: periods.py, aggregate.py, policy.py, models.py
  ports.py         # ActivityWatchGateway, CategorySource, Clock, DayCache
  adapters/        # aw_client_gateway.py, settings_categories.py, file_categories.py
tests/             # unit, property, contract (in-memory Client), golden fixtures
evals/             # synthetic week generator + Inspect AI tasks
docs/              # tutorial, how-to, reference, explanation, adr/
```

## Phases

### Learning week (~1 week)
- [ ] MCP concepts: intro, server concepts, Tools page of 2026-07-28 spec, skim its changelog
- [ ] Python SDK v2: Get started → First steps → Testing → Structured output
- [ ] MCP Inspector: call a tool by hand
- [ ] ActivityWatch data: buckets/events, REST API, "Working with data", categorization; upgrade laptop to v0.14.1
- [ ] "ActivityWatch with agents and AI" guide (read twice; copy its payload example)
- [ ] aw-client: `aw_client/queries.py` — `canonicalEvents()`, `DesktopQueryParams`
- [ ] Tool design & trust: "Writing effective tools for agents", the lethal trifecta post

**Done when:** the SDK example passes a pytest test via the in-memory client, and buckets are visible at `localhost:5600/api/0/buckets/`.

### Phase 1 · Walking skeleton (1 day)
- [ ] uv, src layout, pyproject, ruff, pyright strict, pytest, pre-commit *(scaffold done 2026-10-19)*
- [ ] GitHub Actions: Windows, macOS, Linux × Python 3.11–3.13
- [ ] One tool, `aw_hello` (not "ping" — clashes with the protocol's old ping method)
- [ ] pytest test via the SDK's in-memory `Client`, plus a look in the Inspector
- [ ] Logging to stderr only; write `docs/adr/0001-python-sdk-v2-stdio.md`

**Done when:** CI is green on three OSes with one passing test.

### Phase 2 · Gateway to ActivityWatch (1–2 days)
- [ ] Upgrade laptop to ActivityWatch v0.14.1 first
- [ ] `ActivityWatchGateway` interface + aw-client implementation. Config: host, port, optional API key, testing mode (5666)
- [ ] Tool `aw_status`: server version, devices (incl. synced), bucket names/types, whether categories are on the server
- [ ] Timeouts; errors that tell the *model* what to do ("ActivityWatch isn't running. Ask the user to start it.")
- [ ] `FakeGateway` for tests, fed from JSON fixtures

**Done when:** status shows the real setup, quitting ActivityWatch gives a friendly error, tests run with no ActivityWatch.

### Phase 3 · Summary tools, the core (1–2 weeks)
- [ ] Pure domain core (no I/O): periods, aggregation, comparison
- [ ] Timezone-aware periods (user is in IST), respect start-of-day setting, survive DST elsewhere
- [ ] Tools: `aw_time_by_app`, `aw_time_by_category`, `aw_daily_summary`, `aw_compare_periods` — read-only, structured output
- [ ] Friendly periods: today, yesterday, this week, last 7 days, ISO dates. Cap ranges (92 days), lists (default 10, max 50)
- [ ] Top-N plus "Other" so parts add up to the total
- [ ] Cache finished days
- [ ] Golden tests (3 real days frozen as fixtures vs web UI); Hypothesis property tests (totals add up, no negative durations, order-independent)

**Done when:** results match the web UI within 1 minute on 3 days, including one crossing midnight.

### Phase 4 · Categories (2 days)
- [ ] Read categories from server settings on v0.14 (confirm key via `GET /api/0/settings`); fall back to exported JSON for v0.13.x
- [ ] Respect the active category *set*; pass rules into the canonical query (don't reimplement regex)
- [ ] Resource `aw://categories`: names only, not regex rules
- [ ] Tool `aw_uncategorized`: top uncategorized *apps* (not titles) by time

**Done when:** `time_by_category` uses the user's categories, and the uncategorized report helps write two new rules.

### Phase 5 · Privacy by construction (3–4 days)
- [ ] Layer 1 · Types: no title/URL fields in default outputs (from Phase 3)
- [ ] Layer 2 · Policy: `privacy.toml` deny-list folded into "Private"; items < ~2 min fold into "Other"
- [ ] Layer 3 · Egress guard middleware: scans serialized results for any title/URL from the query; fails closed
- [ ] Layer 4 · Consent: opt-in `aw_top_titles` (off by default; domains only, truncated, marked untrusted, elicitation per request)
- [ ] Layer 5 · Receipts: local JSONL audit log; resource `aw://audit/today`
- [ ] Layer 6 · Quiet telemetry: no OTel export unless opted in; spans never carry payloads
- [ ] `THREAT_MODEL.md`: lethal trifecta mapped
- [ ] README section "What this server shares": every tool and every field

**Done when:** a canary test plants `SECRET-CANARY-7f3a` + a URL, Hypothesis calls every tool with thousands of random inputs, and the canary never appears in default output.

### Phase 6 · Prompts, resources, one "wow" (2–3 days)
- [ ] Prompt **Weekly review**
- [ ] Prompt **Focus check**: context switches/hour, longest focus block, fragmentation score
- [ ] Prompt **Standup note** (from the AI guide's examples)
- [ ] Resources: `aw://categories`, `aw://devices`, `aw://audit/today`
- [ ] Stretch: MCP App with an interactive day timeline (check client support first)

**Done when:** each prompt gives a sensible result on real data in two different clients.

### Phase 7 · Real AI apps and evals (3–4 days)
- [ ] Gemini CLI extension: `gemini-extension.json` + `GEMINI.md`
- [ ] Connect Claude Desktop and VS Code; try the same questions
- [ ] Seeded synthetic-week generator ("student developer") into a test aw-server
- [ ] Inspect AI eval suite (~25 questions): tool choice, numeric accuracy (±2 min), privacy (canary absent), tokens; 2–3 models incl. a small local one
- [ ] Use failures to rewrite tool names/descriptions; keep before/after numbers

**Done when:** README has an eval table and a demo GIF of "How much did I code this week?".

### Phase 8 · Ship it (1 week)
- [ ] Test pyramid: core unit + property (≥90% core coverage), in-memory contract tests, Inspector CLI smoke test in CI, nightly run vs real aw-server (Python + Rust)
- [ ] Diátaxis docs: 5-min tutorial, how-tos per client, generated tool reference, privacy explanation
- [ ] `SECURITY.md`, `PRIVACY.md`, `CHANGELOG.md`, 4–6 ADRs
- [ ] Release: PyPI trusted publishing (`uvx aw-mcp`), MCP Registry via GitHub Actions, `.mcpb` bundle, Gemini extension from repo URL; pin versions
- [ ] Upstream: PR to ActivityWatch's agents-and-AI page, awesome-activitywatch, forum post, blog post; credit 8bitgentleman/activitywatch-mcp-server in README

**Done when:** a stranger installs it with one command in < 2 minutes, and an ActivityWatch maintainer links to it.

## Progress tracker

| Phase | Done when | Status | Date |
|---|---|---|---|
| Learning week | SDK example passes a pytest test; buckets visible | ☐ | |
| 1 · Walking skeleton | CI green on 3 OSes | ☐ | |
| 2 · Gateway | status works; friendly error when AW is off | ☐ | |
| 3 · Summary tools | Matches web UI within 1 min on 3 days | ☐ | |
| 4 · Categories | Uses your categories; 2 new rules written | ☐ | |
| 5 · Privacy by construction | Canary never leaks across thousands of runs | ☐ | |
| 6 · Prompts & resources | Sensible results in 2 clients | ☐ | |
| 7 · Clients & evals | Eval table + demo GIF in README | ☐ | |
| 8 · Ship | Stranger installs in < 2 min; upstream link | ☐ | |
