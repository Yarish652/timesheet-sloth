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

**Full real history (aggregate only).** Export of everything since 2026-07-17: 82 active days,
151 active hours, 19,919 events, 603 blocks ≥ 1 min. Median 1.8 active h/day (IQR 0.9–2.6,
max 7.8). That's a light ActivityWatch footprint, much sparser than the synthetic persona's
~7 h/day, so real-data TabPFN runs will train on a few hundred rows at most. Fine for TabPFN
(it's built for small tables) but the forecast backtest will be noisy. The 365-day export took
~10 min, mostly querying empty days one at a time.

**Categories file, take two.** The first copy of the web UI export produced a 0-byte
`categories.json` → `JSONDecodeError: Expecting value: line 1 column 1`. Copying the real
download (`aw-category-export.json`, 5 KB) fixed it. 16 categories, 13 regex rules. They're the
ActivityWatch *defaults*, never customised. TODO: give the exporter a friendly error for an
empty/invalid file.

**Real data with default rules (aggregate only).** 120-day re-export (4.5 min): 151 h, 19,978
events, 604 blocks. Still 82.9% Uncategorized by time. By block top category: Uncategorized 506,
Media > Video 72, Work > Programming > ActivityWatch 16, Comms > IM 7, Media > Social Media 3.
By app, the uncategorized time is mostly chrome.exe (90.7 h) and **Code.exe (27.5 h)**: the
default Programming rule doesn't match VS Code at all. This is the core weakness of "learn from
your rules": with default rules there are 98 labelled blocks, 4 classes, and one class (Video) is
73% of them. A labeller trained on that will mostly say "Video". The honest fix is better seeds,
not model tuning: a couple of real rules plus hand labels.

**TabPFN weights.** `sloth setup` downloaded one 334 MB file,
`tabpfn-v3.5-fast-20260909.safetensors`, into `%APPDATA%\tabpfn` at ~2.3 MB/s. One checkpoint
appears to serve both classifier and regressor. HF warned about unauthenticated requests; that's
harmless for a one-time download.

## M2: TabPFN labelling (2026-10-23)

**Decision: sloth labels the hours, the user doesn't write rules.** The user didn't want to write
rules by hand ("let this be the first thing it does"). So `sloth/seeds.py` ships a *starter rule
pack*: generic regexes for common apps and sites (VS Code, terminals, GitHub, localhost,
AI chat sites, YouTube, WhatsApp…), written without looking at any titles, using category names
from ActivityWatch's default tree. They only touch events the user's rules left Uncategorized
and are tracked as `label_source`. On real data they cut Uncategorized from 82.9% → 35.6% of
time. Per-pattern aggregate check: VS Code 27.5 h, chatgpt.com 16.7 h, claude.ai 8.5 h,
gemini 5.5 h, localhost 4.4 h, github 2.5 h, … Judgement call worth flagging in the post:
**AI chat sites are counted as Work.** Starter rules default off for synthetic data, whose
fake category tree differs.

**First real TabPFN numbers.** Train/test is time-based: the last 14 days are the test set.
- Synthetic, holdout of categorized blocks: TabPFN 0.995 acc / 0.996 F1 vs top_app lookup 0.789.
- Synthetic, *Uncategorized blocks vs hidden truth*: TabPFN 0.869 / 0.753 vs lookup 0.246 / 0.273.
- Real, starter rules on: TabPFN 0.808 / 0.731 (281 train, 99 test) vs lookup 0.333 / 0.165,
  majority 0.273 / 0.071.
- Real, user rules only: TabPFN 0.727 / 0.336 (48 train, 55 test, 4 classes) vs both baselines
  0.673 / 0.201. Barely better: with 48 rows and one dominant class there's little to learn.

**Is >95% a red flag?** On synthetic data the holdout is 99.5%, and that *is* mostly TabPFN
relearning the regex rules. The test blocks are, by construction, ones a rule matched, so
the matched token is right there in the `tok_*` features. That's why the synthetic "truth" score
on genuinely uncategorized blocks drops to 86.9%: those blocks contain apps and sites the rules
never saw, and the model has to generalise from context (hour, neighbours, shared tokens like
project names). Real-data holdout is 80.8%, not >95%, partly because the real test set mixes in
the vague "Work" class. Neither holdout number is the honest one; hand-tagged blocks are.

**Hand tags: sloth hides its own guess.** `sloth label --tag` asks about the blocks TabPFN is
*least* confident about (active learning: the lazy person answers only the hard ones). First
draft showed "sloth guesses: X, Enter to accept". Removed it: anchoring the human on the model's
answer would inflate exactly the metric the tags exist to measure. Titles are shown only in the
user's terminal; labels.csv stores (block_start, category) only.

**Suspicious real-data output.** `sloth label --dry-run` on real data puts 197 of 272
Uncategorized blocks into plain "Work", at a mean confidence of 0.61. The training set is
dominated by Work/Work > Programming (266 of 380 labelled blocks, mostly from starter rules), so
"Work" is the prior. Some of it is probably right and some isn't. Not tuning this; hand tags
will say.

**Timing surprise: TabPFN `fit` is ~0.3 s, `predict` is 3–13 s.** It's an in-context learner,
so "fit" just preprocesses, and the training rows are processed alongside the test rows at
predict time. CPU, Ryzen 5 7520U: 48 train → 55 test 2.8 s; 281 → 99 8.1 s; 380 → 272 12.8 s;
431 → 130 12.2 s. The whole `sloth eval label` on 120 days of real data runs in ~22 s.

**Test hygiene.** Tests were appending to `reports/timings.jsonl`, the file the write-up cites.
`tests/conftest.py` now points the log at a temp dir. Plumbing tests use a RandomForest stand-in
for speed; one `@pytest.mark.slow` test runs the real TabPFN checkpoint.
