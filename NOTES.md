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

## M3: TabPFN focus forecast (2026-10-23)

**Setup.** TabPFNRegressor (3.5 Fast) predicts focus minutes for each of tomorrow's 48 blocks.
The features are only things known the evening before: block/hour/weekday, same block yesterday,
same block last week, the 4-week same-weekday mean, yesterday's total, and yesterday's/last week's
active minutes. `predict(output_type="main", quantiles=[.1,.5,.9])` returns the mean and the
quantiles in one call, half the cost of calling it twice. The backtest uses a rolling origin:
for each of the last 14 complete days, train only on blocks before that day's 00:00, with
up to 5,000 rows.

**Leakage test that can fail.** `tests/test_no_leakage.py` scrambles every event at or after
D 00:00 (drops 40%, shuffles categories and apps, moves and resizes events, adds fake 25-minute
Code.exe events) and asserts that day D's features, and all earlier features and targets, are
bit-identical. To prove the test has teeth, I planted three leaky features in turn
(`prev_focus_min`, same-day `focus_min`, same-day `active_min`). Each one made it fail.

**Honest result: TabPFN does not win the headline metric.**
- Real data (mean focus **0.40 h/day**): block MAE TabPFN 0.67 min vs always-zero **0.50**,
  4-week mean 0.86, last week 1.03. On **daily totals** TabPFN *is* best: 0.32 h vs zero 0.40,
  4-week 0.46, last week 0.60. Per day, TabPFN never has the lowest block MAE (0/14).
  Always-zero wins 13/14.
- Synthetic (1.46 h/day): block MAE zero 1.82 < TabPFN 2.40 < last week 2.54 < 4-week 2.64.
  Daily totals: 4-week mean 0.54 h best, last week 0.65, TabPFN 0.77, zero 1.46.
- **"Always zero" was added after the first real run**, when the arithmetic (0.40 h × 60 / 48
  blocks = 0.50 min) showed it would beat everything on block MAE. MAE rewards the median, and
  most blocks have zero focus. The predictor was not changed; the baseline is there so the
  table can't flatter it.
- TabPFN's mean underpredicts big days: real 10-09 1.40 h actual vs 0.23 h predicted;
  synthetic 10-22 2.82 h vs 1.66 h. It hedges towards the many zero blocks.
- 80% intervals are **too wide**: they covered 92.7% (real) and 91.2% (synthetic) of blocks,
  more than the nominal 80%. They're conservative, not well calibrated.

**What that means for the product.** At block level, "you probably won't focus at 3 a.m." is
all any method knows, and a zero predictor knows it best. TabPFN's useful outputs are the daily
total (best of four methods on real data) and the per-block distribution: the 90% quantile
marks the hours where focus is *possible*. `plan-tomorrow` uses those. Not tuned, per the brief.
Ideas *not* tried: forecasting daily totals directly, or a two-stage "will I be at the laptop"
plus "how much focus" model.

**Timing.** Real backtest: 910–954 s for 14 days (~65 s per test day, ~5,000 training rows,
48 test rows). Synthetic: 177 s (~13 s per day, ~1,300 rows). TabPFN's predict cost grows with
training rows, since it's an in-context learner. `plan-tomorrow` (one fit) takes ~37 s on synthetic.

**Small things that broke.** Python on Windows defaults to cp1252 for `read_text()`, so a
patch that tried to match a "·" failed (the file round-tripped fine, the match didn't); always
pass `encoding="utf-8"`. Rich wrapped the date columns into unreadable stubs at 80 columns; it
now uses `MM-DD` and `no_wrap`. TabPFN prints "Consider using a GPU…" on every CPU fit; the CLI
filters that one UserWarning.

## Focus definition change (2026-10-24, after M3's eval)

**Why.** M3's rule ("≤ 6 app switches per 30-min block") threw away 66% of real work minutes. The
real workflow is VS Code ⇄ Chrome (docs, ChatGPT, Claude), and every one of those switches
counted as distraction. Two looks at prior art: **RescueTime** counts a "shift" only when you go
from work to communication or personal activity, and **Brown et al. 2023** (Google,
arXiv 2304.04711) detect focus as runs of *related* events, not few app changes. Rize counts raw
switches and reports one every ~35 s, which shows how noisy that measure is.

**New rule (default `focus_rule = "shifts"`, `focus_max_shifts = 2`).** A block's work minutes
count as focus if it has at most 2 shifts from a work category to a known non-work category.
Work-to-work switching is free; Uncategorized and away time are neutral (skipped), so
work → ? → YouTube still counts as one shift. The old rule stays available as
`focus_rule = "switches"`.

**This changed after seeing results, so both are reported.** `sloth eval focus-definition`
prints the sensitivity table, now in eval.md. Real data:
switches ≤ 6 keeps 34% of work minutes (median 0.12 focus h/day); shifts ≤ 2 keeps 89% (median
0.61 h/day); all work time is 0.78 h/day. The curve flattens past 2–3 shifts (94–97%), so the
threshold isn't doing much work.

**Forecast re-measured (real, 14-day rolling origin, mean focus now 0.93 h/day):**
- block MAE: always-zero 1.16 < TabPFN 1.54 < 4-week 1.67 < last week 1.86. Zero still wins
  13/14 days.
- daily-total MAE: 4-week mean **0.65** ≈ TabPFN 0.66 < last week 0.89 < zero 0.93. With the
  old rule TabPFN was best on daily totals (0.32 vs 0.40–0.60); **with the new rule it's a tie
  with the 4-week mean, not a win.**
- 80% interval coverage 88.8%: still too wide, less so than before (92.7%).
- Synthetic: block MAE zero 2.61 < TabPFN 3.09 ≈ 4-week 3.11; daily-total 4-week 0.77 < TabPFN
  0.87 < last week 0.89.
- Runtime 951 s real / 185 s synthetic.

**Reading.** A fairer focus measure doesn't rescue block-level forecasting: the user's focus
still isn't on a schedule. That's the case for M4's switch from "predict when" to "which
starting conditions lead to focus".

## M4: sessions, focus conditions, Gemma agent (2026-10-24, in progress)

**Sessions.** A gap of 15+ min (AFK-filtered) starts a new session; "focused" = ≥ 25 focus minutes
(a Pomodoro). Real data: 295 sessions, 19% focused. Synthetic: 200, 28%.

**Two leaks caught before any number was trusted.**
1. A session's focus was first computed from whole 30-min blocks, and a block can hold the end of
   one session and the start of the next. Now the focus rule is applied per (session, block).
2. `starter_family` ("how did it start") took the first categorized event from *anywhere* in the
   session. "Started with Uncategorized" then meant "no categorized time at all", so it was 0% of
   101 sessions by construction. The scrambling test missed it because scrambling categories
   rarely flips that edge case. Now it's the first 10 minutes only, with a regression test and a
   test that scrambles everything after minute 10 and asserts starter_family doesn't move. After
   the fix: 3% of 127 sessions, low but not zero by definition.

**Focus conditions: TabPFN does beat both baselines here.** P(session becomes focused | start
context), time-based holdout (last 14 days):
- Real: ROC-AUC **0.747** vs rate-by-start-hour 0.516, base rate 0.500; Brier 0.119 vs 0.127 / 0.126;
  log-loss 0.387 vs 0.425 / 0.421. 239 train / 56 test sessions, but only ~8 positives in test
  (14% focused), so the AUC is noisy. Calibration gains are small.
- Synthetic: AUC 0.804 vs 0.700 / 0.500; Brier 0.180 vs 0.198 / 0.228.
- First time in this project TabPFN clearly wins: it's the right question for the data.

**The real suggestion is partly circular.** "In the evening, start with Work instead of an
Uncategorized app: 7% → 44% (39 sessions)." Starting with work means you're already working.
It's still actionable for the lazy version ("open VS Code before Chrome"), but it isn't a deep
insight, and it's correlational. Real routine score 0.33 → "design mode"; synthetic 0.57 → routine.

**TabPFN is slow on tiny tables with categoricals.** 239 rows × 9 features: fit 22 s, predict
30–41 s even for 10 rows. Synthetic check: auto n_estimators resolves to 4. Declaring the 3
categorical columns roughly doubles predict time (37 s vs 15 s). Kept them declared because
they *are* categories.

**`build_sessions` was 8 s per call** (thousands of small pandas lookups); grouped aggregations
bring it to 0.7 s with identical output. The test file went from 254 s to ~70 s.

**Gemma, first real runs (CPU, 13.8 GB RAM, ~1.3 GB free).**
- `standup --date 2026-10-21 --trace`: 2 tool calls (day_summary, focus_conditions), 3 LLM calls,
  474 s of LLM (first call 240 s, 124 s of it model load; then 52 s; then 182 s for 154 output
  tokens), **13 min 43 s total**. Number check passed. Prose used only tool numbers, said
  "tends to" for the suggestion.
- `timesheet --week --trace`: 2 tool calls, 2 LLM calls, 553 s. Load 113 s (reload after the context
  change), then 392 tokens in 326 s (~1.2 tok/s vs ~7 tok/s in the first standalone test).
- **The number checker flagged "5.3"**, and the bug was in the checker: the tool said 5.25, Gemma
  rounded half-up, and the checker used Python's round-half-to-even (5.2). Now it accepts any
  number within half a unit of the prose's precision (5.25 → 5.3 or 5.2 ok; 5.4 flagged).
- Memory: Gemma 4's default context is 131,072 tokens; capped at `llm_num_ctx = 8192`. A
  3.6 GB `java` process (not ours) plus Gemma (3.2 GB) plus TabPFN left ~1.3 GB free.
- **`plan-tomorrow` was killed by Claude Code's low-memory safeguard** mid-TabPFN, while Gemma sat
  in RAM from the previous command (`keep_alive` 30 m, my change). Fix: TabPFN-backed tools are
  prefetched and cached *before* the first LLM call, and `keep_alive` is now 2 m, so the two
  models don't share RAM within or across commands.

**Dashboard comparison caveat.** ActivityWatch's web UI starts the day at 04:00 by default; sloth
uses midnight. On 2026-10-21 (first activity 00:03) that's 4.14 h (sloth) vs 2.10 h (04:00
boundary). Candidate fix: a `day_start_hour` setting; not done because it shifts every date-based
number from M2/M3.

**Privacy path.** Tools return aggregates (category names, exe names, counts). The only place a
tool result enters the message list is `agent._tool_message`, which serialises through
`privacy.Guard.check`. That raises (without echoing the match) if any title ≥ 6 chars (single plain
words need ≥ 12) or URL/host from the data appears. A canary test plants `SECRET-CANARY-7f3a` in
titles and URLs and asserts that no tool output contains it, and that the guard blocks it.

## M5: README, verification, polish (2026-10-25)

**M4 verified after the fact.** Full suite on the committed M4 tree: 69 passed in 261 s (my M4
report said 70; I'd miscounted 52 + 18 instead of 51 + 18).

**`plan-tomorrow` with Gemma, real data, after closing a 3.6 GB Java process (4.1 GB free):**
5 min 6 s end to end. TabPFN forecast 116 s (4,704 training blocks → 48), focus conditions
fit 0.2 s + predict **4.2 s** (the same call took 30–41 s with ~1 GB free, so memory, not
TabPFN, caused the earlier slowness), Gemma 2 calls 155 s (57 s model load, 155 output tokens at
~2.5 tok/s). Number check passed. Two flaws: it forecast 2026-10-23 because the export ended
mid-day on 10-23 and "tomorrow" is relative to the data; and Gemma dropped the "39 sessions"
evidence the system prompt asks for. The checker catches invented numbers, not missing ones.
Best windows were 00:00–03:00 at ~3–4 expected minutes each: honest given this person's
nights, but hardly a plan.

**Offline.** `uv run --offline sloth --help` works. Every real-data command in M2–M4 ran behind
netguard, so any non-local connection attempt would have raised; none did.

**Fresh clone, exactly the README steps** (cloned from the local repo; GitHub not pushed yet):
`uv sync` 22 s · `ollama pull` (cached) · `sloth setup` 42 s (weights already in the TabPFN cache,
so no download) · `python -m sloth.export` 4 min 43 s (84 active days, 153 h, 617 blocks; no
`categories.json` in the clone → 0 rules → starter rules only). Then `sloth label --dry-run`
worked, and **69 tests passed in the clone** (124 s), so the suite doesn't lean on local files.
The clone held a copy of the real export and was deleted afterwards.
