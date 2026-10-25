---
title: "Timesheet Sloth: I asked TabPFN to predict when I'd focus. It said 'never'. So I made it lazy instead."
published: false
tags: devchallenge, opensource, ai, productivity
---

*This is a submission for the Hacktoberfest Open-Source AI Challenge, Week 3: Slow and Steady.*

## What I built

**Timesheet Sloth** is a fully local agent that reads my [ActivityWatch](https://activitywatch.net)
history and does the work I hate:

- writes my **daily standup** and **weekly timesheet**,
- **labels the time my category rules missed** (82.9% of my hours were "Uncategorized"),
- and, while my laptop is idle, prepares **one suggestion** for when I sit back down: what to
  open first, and what I was last working on.

Two open models do the work: **TabPFN 3.5** (a tabular foundation model, open weights, CPU) for all
the numbers, and **Gemma 4 E4B** via Ollama (native tool calling) for the prose. **Nothing leaves
my machine.** The only network calls are to `localhost:5600` and `localhost:11434`, and that's
enforced in code.

Repo: https://github.com/Yarish652/MCP-AW

<!-- DEMO GIF HERE: sloth next -> toast -> sloth standup --trace (synthetic data) -->

## Why it's lazy (in a good way)

The laziest possible productivity tool is one you never have to open. Sloth waits until
ActivityWatch says I've been away for 10 minutes, then does the slow part: an incremental export,
TabPFN scoring of every way I've ever started a session for each of the next 24 hours, and Gemma
rewording the top suggestion. When I come back after a break, a desktop notification is already
waiting, for example:

> *Evening sessions that start with Work tend to become focused 36% of the time (from 40 sessions),
> vs 4% for your usual Uncategorized start. Pick up where you left off: Work in chrome.exe.*

That's the real text from my first real cycle: I stepped away, the daemon built the cache while
I was gone, and the notification was waiting when I came back 21 minutes later.

Building that cache takes 95 seconds. Reading it takes 0.07 seconds.

## The honest part: the first idea failed

The original plan was a forecaster: "you'll focus best tomorrow 10:00–11:30". I built it properly:
TabPFN regressor, day-ahead features only, a 14-day rolling-origin backtest, and a test that
scrambles every event after midnight and asserts the features don't change. Results on my real
data (14 weeks, 151 active hours):

| forecast of focus minutes | block MAE (min) | daily-total MAE (h) |
|---|---|---|
| TabPFN | 1.54 | 0.66 |
| same as last week | 1.86 | 0.89 |
| 4-week rolling mean | 1.67 | **0.65** |
| **always predict zero** | **1.16** | 0.93 |

**"Always zero" won.** Not because TabPFN is bad, but because I'm not predictable: only 11–13% of
my focus blocks recur in the same slot a day or a week later, and 41% of my focus happens between
midnight and 6 AM. There's no schedule to forecast.

So I changed the question from *"when will I focus?"* to *"given how a session starts, how likely
is it to become 25+ focused minutes?"* That one is answerable:

| P(session becomes focused) | ROC-AUC | Brier |
|---|---|---|
| **TabPFN** | **0.747** | **0.119** |
| rate by start hour | 0.516 | 0.127 |
| base rate | 0.500 | 0.126 |

The test set only had ~8 focused sessions, so treat that AUC as noisy. And the main finding
("start with work and you're more likely to keep working") is partly circular. But it's a
suggestion I can act on, which a forecast for 3 AM never was.

## How TabPFN is used (three times)

1. **Labelling Uncategorized time.** Blocks my rules (plus a small generic starter pack)
   recognise are training data; TabPFN labels the rest. Window titles never reach the model: they
   become "share of block time containing keyword *k*" features. Accuracy 0.81 vs 0.33 for a
   "most common category for this app" baseline, though part of that is relearning the rules
   (on synthetic data, 99.5% on rule-labelled blocks drops to 87% on truly uncategorized ones).
2. **Forecasting** focus per block, with 10/50/90% quantiles from one `predict(output_type="main")`
   call. Its intervals were too wide (88.8% coverage for a nominal 80%).
3. **Focus conditions**: a classifier over session-start context, plus "what if" scoring of
   alternatives, but only ways of starting I've actually used 8+ times. No extrapolation.

A surprise for the post: TabPFN's `fit` takes ~0.3 s; `predict` does all the work, because it's an
in-context learner. On my Ryzen 5 laptop: ~4 s for the conditions model with RAM to spare, 30–40 s
for the same call when RAM was tight.

## Privacy, by construction

- **No network except localhost.** `netguard.py` patches `socket.connect`/`getaddrinfo` to refuse
  non-loopback addresses for every command except one-time `sloth setup`. (I read TabPFN's source:
  it only contacts Prior Labs/Hugging Face when the weights aren't cached.)
- **No titles or URLs to the LLM.** Tools return aggregates. The single code path into Gemma's
  prompt runs a guard that scans for any title or URL from my data and fails closed. A test plants
  `SECRET-CANARY-7f3a` in titles and URLs and checks it never comes out.
- **No invented numbers.** A checker flags any number in Gemma's text that isn't in a tool result.
  It caught "5.3" on the first timesheet. The bug turned out to be mine: the tool said 5.25, Gemma
  rounded half-up, and my checker used Python's round-half-to-even.
- **The closed model never saw my data either.** Sloth was built with a closed-model coding
  assistant under one rule: never open the real export, only run scripts that print aggregates.
  It worked against a synthetic data generator the whole time.

## Things that broke (my favourite part)

- **Two data leaks**, caught before I trusted a number. A session's focus counted the *next*
  session's events. And "how did this session start?" looked at the *whole* session, which made
  "started Uncategorized → 0% focused" true by definition. Both now have regression tests.
- **My focus metric was wrong.** "≤ 6 app switches per half-hour" threw away 66% of my real work,
  because VS Code ⇄ docs ⇄ ChatGPT counted as distraction. I switched to RescueTime's idea of a
  *shift* (only work → non-work counts) and reported both versions. TabPFN's one forecasting win
  (daily totals) disappeared under the fairer metric. I left that in.
- **The machine ran out of memory.** Gemma (3.2 GB) + TabPFN + an unrelated 3.6 GB Java process on
  16 GB, and Gemma's default 131k-token context. The fix: cap the context, run TabPFN *before*
  Gemma loads, and unload Gemma afterwards. A memory-starved `standup` took 13 min 43 s; afterwards
  `plan-tomorrow` ran in 5 min, and the conditions model went from 30–40 s to 4 s.
- **Gemma wrote "focused 0.41% of the time"** on the first real cycle (it's 41%), and forgot to
  say *what* to open. My checker passed it because it stripped the % sign. Now percentages must
  match fractions × 100, and a rewording that drops the suggestion is thrown away. Since then,
  sloth has rejected every rewording Gemma produced, and shows its own template sentence instead.
  Small local LLMs are great at prose and bad at percentages.
- **My AI pair programmer showed up as my work.** The "resume" suggestion said "Work >
  Programming in WindowsTerminal.exe, 3 min ago". That was the coding assistant's terminal.
  ActivityWatch can't tell us apart.

## Try it

```sh
git clone https://github.com/Yarish652/MCP-AW.git timesheet-sloth && cd timesheet-sloth
uv sync
ollama pull gemma4:e4b
uv run sloth setup
uv run python -m sloth.export
```

No ActivityWatch? `uv run sloth synth` makes six weeks of fake data for every command.

## What's next

Sloth logs every suggestion it shows (no titles). Next: learn which suggestions I actually take,
and run a coin-flip experiment (suggest or don't, at random) to find out whether they *help*,
rather than just correlate with good sessions. Patterns from one person's history aren't causes,
and the post would be lying if it said otherwise.

*Slow and steady. Mostly slow.* 🦥
