# Build notes

Running log of decisions, surprises and breakage. Newest at the bottom of each day.

## 2026-10-22 (Thu)

**Environment.** Windows 11, Ryzen 5 7520U (4C/8T), 13.8 GB usable RAM, no GPU. System Python is
3.14, so the project pins a 3.13 venv via uv. `tabpfn` 9.2.0 resolves with `torch` 2.14.1+cpu
(PyPI's Windows torch wheels are CPU-only, so there's no CUDA download to fight). ActivityWatch
v0.13.2 on :5600 with window, AFK and web buckets.

**Repo.** Replaced the `aw-mcp` scaffold (an MCP server for ActivityWatch I'd started) with this
project. The MCP idea survives as the stretch goal: the agent's tools are written once in
`tools.py` and can be served over MCP as well as to Gemma. License switched MPL-2.0 → MIT.

**TabPFN 3.5 Fast is there.** `tabpfn.constants.ModelVersion` lists `v3.5-fast`, built with
`TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)`. `TabPFNRegressor.predict`
takes `output_type=` and `quantiles=`, so forecast intervals come straight from the model.

**TabPFN and the network.** Read the package source before trusting "local": the gated 3.5
weights need a one-time browser license acceptance, and the token is verified against PriorLabs'
API, but only inside `_download_model`, i.e. when the weights aren't cached yet. After that,
nothing. No telemetry module in the package. To make "only localhost at runtime" a guarantee
rather than a promise, `sloth/netguard.py` monkeypatches `socket.connect`/`getaddrinfo` to refuse
any non-loopback address, and every CLI command except `sloth setup` installs it first.

**v0.13 categories aren't on the server.** `GET /api/0/settings` returns `{}`; on v0.13 the
category rules live in the web UI's browser storage. The exporter has to get them some other way.

**Design: blocks come from events.** `features.build_blocks` rebuilds 30-minute blocks from
`events.csv`, so synthetic and real data share one code path and the event file is the only
contract with the exporter.

**Design: titles never reach a model.** Titles are reduced to keyword tokens, and a block keeps
only "share of active time whose title/URL contains token *t*" for a ~40-token vocabulary picked
from training blocks. Feature columns are named `tok_0..tok_39`, so even debug output shows no words.

**Synthetic persona.** A student developer in IST, 42 days, ~10k events, ~8.8 active h/day. Rule
coverage is tuned so ~26% of active time is Uncategorized (PyCharm, WhatsApp, Zoom, Notion,
Stack Overflow, localhost, arXiv, X aren't matched by the fake rules). Every synthetic event keeps
a `true_category`, which lets us score the labeller on *genuinely* uncategorized time.

**A bias to watch in the labeller.** Training blocks are, by construction, the ones the regex
rules recognise, and the blocks we predict are the ones they don't. So a holdout split of
categorized blocks will look better than performance on real Uncategorized time. The synthetic
`true_category` measures that gap; on real data, `labels.csv` hand tags are the honest check.

## 2026-10-23 (Fri)

**Exporter moved in.** `sloth_export.py` → `src/sloth/export.py` (`python -m sloth.export`).
Behaviour kept: same AW query, same URL borrowing from the web bucket, same CSV columns, same
block aggregation. Changes: the writers are split into functions so `sloth.synthetic` reuses
them byte-for-byte; `--out` defaults to `$SLOTH_DATA_DIR` (or `./data`) and is created if missing;
and a new `--categories` fallback (below).

**Surprise: the exporter found 0 category rules on the real server.** It reads
`/api/0/settings/` and that's `{}` on v0.13.2, so all 19.8 real hours in the first 7-day export
came out 100% Uncategorized: nothing to train on. Fix: `--categories <file>` (default
`<out>/categories.json`) reads the web UI's export (Settings → Categorization → Export, format
`{"id": ..., "categories": [{name: [...], rule: {type, regex, ignore_case}}]}`). Server rules
still win when present (v0.14+).

**Real data volume.** Buckets start 2026-07-17, so ~14 weeks of history. The first 7-day export:
8 active days, 20 active hours, 4,150 events, 77 blocks ≥ 1 min. That's far below TabPFN's
~5k-row CPU limit, so no subsampling needed. The worry flips: there may be too *few* rows.

**Two block definitions exist, on purpose.** `blocks.csv` is the exporter's original output
(switches counted per event-start block, blocks < 1 min dropped). The models use
`features.build_blocks`, which rebuilds blocks from `events.csv` on a full grid (idle blocks
included, for lag features) and counts switches *within* a block. Kept the exporter's file as-is
since "keep its behavior" was the brief; the model features are a superset.

**Synthetic tuning.** Fake rules now leave VLC, PyCharm, WhatsApp, Zoom, Notion, Stack Overflow,
localhost, arXiv and X uncovered → 29.8% Uncategorized over 28 days. Uncovering *both* social
sites made "Media > Social" vanish from the labelled data entirely (36.6% Uncategorized): a class
the rules never match can't be learned. Real-data analogue: if a whole category of yours has no
working rule, the labeller can't invent it. There's a test for this now. Weekdays vary by design
(Wednesday lab, Friday evening out, lighter weekends): 5.4–8.1 active h/day.

**Gemma 4 E4B over Ollama.** Ollama 0.40.8, model 6.6 GB. A single tool-calling round trip:
57.5 s total, 19.9 s of it model load, 264 output tokens for a one-line tool call (looks like
thinking tokens; try `think: false`). Native tool calling works: it returned
`day_summary(date="2026-10-21")` correctly on the first try.
