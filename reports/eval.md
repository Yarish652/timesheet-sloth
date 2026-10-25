# Evaluation

Numbers written by `sloth eval`. Each section is replaced when its command is re-run.
Real-data sections are computed locally and contain aggregates only.

<!-- summary -->
## Summary (real data, final)

| Task | TabPFN | Best baseline | Did TabPFN win? |
|---|---|---|---|
| Label Uncategorized (accuracy, holdout, starter rules on) | 0.808 | top_app lookup 0.333 | Yes. Partly relearning the rules; no hand-labelled score yet |
| Label Uncategorized (accuracy, your rules only) | 0.727 | majority / lookup 0.673 | Barely (48 training blocks) |
| Forecast focus, block MAE (min, shifts<=2) | 1.54 | always zero **1.16** | **No** |
| Forecast focus, daily-total MAE (h, shifts<=2) | 0.66 | 4-week mean **0.65** | **No** (tie) |
| Forecast focus, daily-total MAE (h, old switches<=6 rule) | **0.32** | always zero 0.40 | Yes, under the old focus rule only |
| Session becomes focused (ROC-AUC, holdout) | **0.747** | rate by start hour 0.516 | Yes. Only ~8 positives in test |
| Session becomes focused (Brier) | **0.119** | base rate 0.126 | Yes, small |

Details for each row are in the sections below.
<!-- /summary -->

<!-- label:synthetic data, starter rules off -->
## Labeller (synthetic data, starter rules off)

Time-based holdout: train 2026-09-25..2026-10-08, test 2026-10-09..2026-10-22, 7 classes. Test blocks are ones the rules categorized, so this measures how well TabPFN relearns the rules.

| method | accuracy | macro-F1 | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|
| TabPFN | 0.995 | 0.996 | 222 | 209 | 13.391 |
| majority class | 0.287 | 0.064 | 222 | 209 | 0.001 |
| top_app lookup | 0.789 | 0.749 | 222 | 209 | 0.005 |

Uncategorized blocks scored against the generator's hidden true category (the case that matters: time the rules missed).

| method | accuracy | macro-F1 | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|
| TabPFN | 0.869 | 0.753 | 431 | 130 | 12.438 |
| majority class | 0.185 | 0.052 | 431 | 130 | 0.000 |
| top_app lookup | 0.246 | 0.273 | 431 | 130 | 0.004 |

Leakage check over 54 feature columns: PASS: no feature is the block's category, a rule output, or a raw title/URL.

_Updated 2026-10-23 02:03._
<!-- /label:synthetic data, starter rules off -->

<!-- label:real data, starter rules on -->
## Labeller (real data, starter rules on)

Time-based holdout: train 2026-07-17..2026-10-09, test 2026-10-10..2026-10-23, 7 classes. Test blocks are ones the rules categorized, so this measures how well TabPFN relearns the rules.

| method | accuracy | macro-F1 | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|
| TabPFN | 0.808 | 0.731 | 281 | 99 | 12.795 |
| majority class | 0.273 | 0.071 | 281 | 99 | 0.001 |
| top_app lookup | 0.333 | 0.165 | 281 | 99 | 0.004 |

Leakage check over 54 feature columns: PASS: no feature is the block's category, a rule output, or a raw title/URL.

_Updated 2026-10-23 02:04._
<!-- /label:real data, starter rules on -->

<!-- label:real data, starter rules off -->
## Labeller (real data, starter rules off)

Time-based holdout: train 2026-07-19..2026-10-08, test 2026-10-10..2026-10-23, 4 classes. Test blocks are ones the rules categorized, so this measures how well TabPFN relearns the rules.

| method | accuracy | macro-F1 | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|
| TabPFN | 0.727 | 0.336 | 48 | 55 | 7.304 |
| majority class | 0.673 | 0.201 | 48 | 55 | 0.001 |
| top_app lookup | 0.673 | 0.201 | 48 | 55 | 0.004 |

Leakage check over 54 feature columns: PASS: no feature is the block's category, a rule output, or a raw title/URL.

_Updated 2026-10-23 02:04._
<!-- /label:real data, starter rules off -->

<!-- forecast:real data, starter rules on, focus=switches<=6 -->
## Focus forecast backtest (real data, starter rules on, focus=switches<=6)

Rolling origin over the last 14 complete days: each test day is predicted by a model trained only on blocks before that day's 00:00. Focus = minutes in Work categories within blocks with <= 6 app switches. Mean actual focus 0.40 h/day. Runtime 910 s.

| method | block MAE (min) | daily-total MAE (h) | days best |
|---|---|---|---|
| TabPFN | 0.668 | 0.321 | 0 |
| same as last week | 1.032 | 0.604 | 3 |
| rolling 4-week average | 0.862 | 0.463 | 0 |
| always zero | 0.496 | 0.397 | 13 |

TabPFN 80% predictive interval (10%-90% quantiles) covered 92.7% of blocks.

<details><summary>Per day</summary>

| test day | trained on | actual h | TabPFN h | TabPFN MAE | last-week MAE | 4-wk MAE | zero MAE |
|---|---|---|---|---|---|---|---|
| 2026-10-09 | 2026-07-17..2026-10-08 | 1.404 | 0.232 | 1.916 | 2.378 | 1.846 | 1.755 |
| 2026-10-10 | 2026-07-17..2026-10-09 | 1.520 | 0.228 | 2.139 | 2.163 | 2.070 | 1.900 |
| 2026-10-11 | 2026-07-17..2026-10-10 | 0.662 | 0.253 | 1.111 | 0.827 | 1.053 | 0.827 |
| 2026-10-12 | 2026-07-17..2026-10-11 | 0.178 | 0.126 | 0.365 | 0.222 | 0.620 | 0.222 |
| 2026-10-13 | 2026-07-17..2026-10-12 | 0.111 | 0.123 | 0.256 | 0.666 | 0.447 | 0.139 |
| 2026-10-14 | 2026-07-17..2026-10-13 | 0.001 | 0.182 | 0.227 | 0.361 | 0.233 | 0.001 |
| 2026-10-15 | 2026-07-17..2026-10-14 | 0.007 | 0.146 | 0.182 | 1.549 | 1.154 | 0.009 |
| 2026-10-16 | 2026-07-17..2026-10-15 | 0.321 | 0.179 | 0.568 | 1.958 | 0.826 | 0.401 |
| 2026-10-17 | 2026-07-17..2026-10-16 | 0.274 | 0.146 | 0.507 | 2.243 | 0.884 | 0.343 |
| 2026-10-18 | 2026-07-17..2026-10-17 | 0.185 | 0.105 | 0.354 | 0.597 | 0.540 | 0.231 |
| 2026-10-19 | 2026-07-17..2026-10-18 | 0.152 | 0.120 | 0.324 | 0.410 | 0.245 | 0.191 |
| 2026-10-20 | 2026-07-17..2026-10-19 | 0.081 | 0.219 | 0.323 | 0.240 | 0.565 | 0.101 |
| 2026-10-21 | 2026-07-17..2026-10-20 | 0.652 | 0.093 | 0.870 | 0.813 | 0.904 | 0.815 |
| 2026-10-22 | 2026-07-17..2026-10-21 | 0.006 | 0.170 | 0.205 | 0.017 | 0.688 | 0.008 |

</details>

_Updated 2026-10-23 13:22._
<!-- /forecast:real data, starter rules on, focus=switches<=6 -->

<!-- forecast:synthetic data, starter rules off, focus=switches<=6 -->
## Focus forecast backtest (synthetic data, starter rules off, focus=switches<=6)

Rolling origin over the last 14 complete days: each test day is predicted by a model trained only on blocks before that day's 00:00. Focus = minutes in Work categories within blocks with <= 6 app switches. Mean actual focus 1.46 h/day. Runtime 177 s.

| method | block MAE (min) | daily-total MAE (h) | days best |
|---|---|---|---|
| TabPFN | 2.397 | 0.774 | 0 |
| same as last week | 2.541 | 0.648 | 2 |
| rolling 4-week average | 2.641 | 0.542 | 0 |
| always zero | 1.820 | 1.456 | 13 |

TabPFN 80% predictive interval (10%-90% quantiles) covered 91.2% of blocks.

<details><summary>Per day</summary>

| test day | trained on | actual h | TabPFN h | TabPFN MAE | last-week MAE | 4-wk MAE | zero MAE |
|---|---|---|---|---|---|---|---|
| 2026-10-09 | 2026-09-25..2026-10-08 | 0.670 | 1.638 | 2.259 | 2.113 | 2.357 | 0.837 |
| 2026-10-10 | 2026-09-25..2026-10-09 | 0.000 | 0.524 | 0.655 | 0.731 | 0.749 | 0.000 |
| 2026-10-11 | 2026-09-25..2026-10-10 | 0.223 | 0.405 | 0.741 | 0.648 | 0.702 | 0.278 |
| 2026-10-12 | 2026-09-25..2026-10-11 | 1.471 | 1.279 | 2.485 | 3.323 | 3.451 | 1.839 |
| 2026-10-13 | 2026-09-25..2026-10-12 | 2.197 | 1.280 | 3.237 | 3.822 | 3.706 | 2.746 |
| 2026-10-14 | 2026-09-25..2026-10-13 | 0.726 | 1.368 | 2.138 | 2.524 | 2.445 | 0.908 |
| 2026-10-15 | 2026-09-25..2026-10-14 | 2.700 | 1.329 | 3.670 | 3.167 | 3.290 | 3.375 |
| 2026-10-16 | 2026-09-25..2026-10-15 | 1.952 | 1.045 | 2.941 | 2.840 | 3.185 | 2.440 |
| 2026-10-17 | 2026-09-25..2026-10-16 | 0.000 | 0.355 | 0.443 | 0.000 | 0.499 | 0.000 |
| 2026-10-18 | 2026-09-25..2026-10-17 | 0.552 | 0.272 | 0.927 | 0.968 | 1.065 | 0.690 |
| 2026-10-19 | 2026-09-25..2026-10-18 | 1.970 | 1.236 | 2.897 | 2.785 | 3.650 | 2.462 |
| 2026-10-20 | 2026-09-25..2026-10-19 | 2.531 | 1.306 | 3.208 | 3.581 | 3.210 | 3.164 |
| 2026-10-21 | 2026-09-25..2026-10-20 | 2.570 | 1.193 | 3.567 | 3.492 | 3.591 | 3.213 |
| 2026-10-22 | 2026-09-25..2026-10-21 | 2.823 | 1.657 | 4.395 | 5.575 | 5.079 | 3.528 |

</details>

_Updated 2026-10-23 13:25._
<!-- /forecast:synthetic data, starter rules off, focus=switches<=6 -->

<!-- focus-definition:real data, starter rules on -->
## Focus definition sensitivity (real data, starter rules on)

How much of the time in work categories counts as focus under each rule. `switches` is the original M1-M3 rule (any app change). `shifts` counts only moves from work to a known non-work category (RescueTime-style), so editor <-> docs/AI chat is free and Uncategorized is neutral. 82 active days.

| rule | focus h/day (median) | focus h/day (mean) | % of work kept |
|---|---|---|---|
| switches <= 0 | 0.000 | 0.069 | 8% |
| switches <= 2 | 0.011 | 0.170 | 19% |
| switches <= 4 | 0.074 | 0.262 | 29% |
| switches <= 6 | 0.118 | 0.310 | 34% |
| switches <= 8 | 0.171 | 0.373 | 41% |
| switches <= 10 | 0.239 | 0.407 | 45% |
| shifts <= 0 | 0.401 | 0.652 | 72% |
| shifts <= 1 | 0.532 | 0.753 | 83% |
| shifts <= 2  (current) | 0.615 | 0.810 | 89% |
| shifts <= 3 | 0.708 | 0.850 | 94% |
| shifts <= 4 | 0.711 | 0.856 | 95% |
| shifts <= 5 | 0.720 | 0.871 | 96% |
| shifts <= 6 | 0.720 | 0.875 | 97% |
| no rule (all work time) | 0.778 | 0.906 | 100% |

_Updated 2026-10-24 00:37._
<!-- /focus-definition:real data, starter rules on -->

<!-- focus-definition:synthetic data, starter rules off -->
## Focus definition sensitivity (synthetic data, starter rules off)

How much of the time in work categories counts as focus under each rule. `switches` is the original M1-M3 rule (any app change). `shifts` counts only moves from work to a known non-work category (RescueTime-style), so editor <-> docs/AI chat is free and Uncategorized is neutral. 28 active days.

| rule | focus h/day (median) | focus h/day (mean) | % of work kept |
|---|---|---|---|
| switches <= 0 | 0.000 | 0.094 | 4% |
| switches <= 2 | 0.536 | 0.632 | 29% |
| switches <= 4 | 1.118 | 1.131 | 52% |
| switches <= 6 | 1.865 | 1.567 | 72% |
| switches <= 8 | 2.371 | 1.889 | 86% |
| switches <= 10 | 2.586 | 2.083 | 95% |
| shifts <= 0 | 1.549 | 1.394 | 64% |
| shifts <= 1 | 2.384 | 2.007 | 92% |
| shifts <= 2  (current) | 2.712 | 2.167 | 99% |
| shifts <= 3 | 2.761 | 2.178 | 100% |
| shifts <= 4 | 2.761 | 2.185 | 100% |
| shifts <= 5 | 2.761 | 2.185 | 100% |
| shifts <= 6 | 2.761 | 2.185 | 100% |
| no rule (all work time) | 2.761 | 2.185 | 100% |

_Updated 2026-10-24 00:37._
<!-- /focus-definition:synthetic data, starter rules off -->

<!-- forecast:real data, starter rules on, focus=shifts<=2 -->
## Focus forecast backtest (real data, starter rules on, focus=shifts<=2)

Rolling origin over the last 14 complete days: each test day is predicted by a model trained only on blocks before that day's 00:00. Focus = minutes in Work within blocks with <= 2 shifts from work to non-work. Mean actual focus 0.93 h/day. Runtime 951 s.

| method | block MAE (min) | daily-total MAE (h) | days best |
|---|---|---|---|
| TabPFN | 1.543 | 0.663 | 0 |
| same as last week | 1.857 | 0.892 | 3 |
| rolling 4-week average | 1.671 | 0.655 | 0 |
| always zero | 1.164 | 0.931 | 13 |

TabPFN 80% predictive interval (10%-90% quantiles) covered 88.8% of blocks.

<details><summary>Per day</summary>

| test day | trained on | actual h | TabPFN h | TabPFN MAE | last-week MAE | 4-wk MAE | zero MAE |
|---|---|---|---|---|---|---|---|
| 2026-10-09 | 2026-07-17..2026-10-08 | 1.615 | 0.412 | 2.257 | 2.796 | 2.421 | 2.018 |
| 2026-10-10 | 2026-07-17..2026-10-09 | 2.481 | 0.449 | 3.503 | 3.180 | 3.888 | 3.102 |
| 2026-10-11 | 2026-07-17..2026-10-10 | 2.147 | 0.424 | 3.002 | 2.683 | 3.169 | 2.683 |
| 2026-10-12 | 2026-07-17..2026-10-11 | 0.725 | 0.483 | 1.347 | 0.906 | 1.304 | 0.906 |
| 2026-10-13 | 2026-07-17..2026-10-12 | 0.692 | 0.488 | 1.322 | 1.441 | 1.326 | 0.865 |
| 2026-10-14 | 2026-07-17..2026-10-13 | 0.015 | 0.366 | 0.438 | 0.099 | 0.154 | 0.019 |
| 2026-10-15 | 2026-07-17..2026-10-14 | 0.007 | 0.522 | 0.644 | 1.554 | 1.422 | 0.009 |
| 2026-10-16 | 2026-07-17..2026-10-15 | 1.447 | 0.348 | 2.039 | 3.352 | 2.347 | 1.808 |
| 2026-10-17 | 2026-07-17..2026-10-16 | 0.790 | 0.447 | 1.423 | 2.419 | 1.068 | 0.987 |
| 2026-10-18 | 2026-07-17..2026-10-17 | 0.854 | 0.404 | 1.390 | 3.207 | 1.696 | 1.068 |
| 2026-10-19 | 2026-07-17..2026-10-18 | 0.592 | 0.360 | 1.067 | 1.645 | 0.967 | 0.741 |
| 2026-10-20 | 2026-07-17..2026-10-19 | 0.428 | 0.424 | 0.844 | 1.168 | 1.118 | 0.535 |
| 2026-10-21 | 2026-07-17..2026-10-20 | 0.945 | 0.329 | 1.294 | 1.161 | 1.277 | 1.181 |
| 2026-10-22 | 2026-07-17..2026-10-21 | 0.301 | 0.567 | 1.025 | 0.386 | 1.230 | 0.377 |

</details>

_Updated 2026-10-24 00:55._
<!-- /forecast:real data, starter rules on, focus=shifts<=2 -->

<!-- forecast:synthetic data, starter rules off, focus=shifts<=2 -->
## Focus forecast backtest (synthetic data, starter rules off, focus=shifts<=2)

Rolling origin over the last 14 complete days: each test day is predicted by a model trained only on blocks before that day's 00:00. Focus = minutes in Work within blocks with <= 2 shifts from work to non-work. Mean actual focus 2.09 h/day. Runtime 185 s.

| method | block MAE (min) | daily-total MAE (h) | days best |
|---|---|---|---|
| TabPFN | 3.090 | 0.867 | 0 |
| same as last week | 3.194 | 0.889 | 3 |
| rolling 4-week average | 3.112 | 0.767 | 2 |
| always zero | 2.613 | 2.090 | 9 |

TabPFN 80% predictive interval (10%-90% quantiles) covered 90.6% of blocks.

<details><summary>Per day</summary>

| test day | trained on | actual h | TabPFN h | TabPFN MAE | last-week MAE | 4-wk MAE | zero MAE |
|---|---|---|---|---|---|---|---|
| 2026-10-09 | 2026-09-25..2026-10-08 | 1.025 | 2.322 | 3.016 | 3.103 | 3.217 | 1.281 |
| 2026-10-10 | 2026-09-25..2026-10-09 | 0.342 | 0.671 | 1.207 | 1.179 | 1.345 | 0.428 |
| 2026-10-11 | 2026-09-25..2026-10-10 | 0.223 | 0.563 | 0.935 | 0.648 | 0.892 | 0.278 |
| 2026-10-12 | 2026-09-25..2026-10-11 | 2.485 | 2.355 | 3.546 | 3.993 | 3.963 | 3.106 |
| 2026-10-13 | 2026-09-25..2026-10-12 | 3.214 | 2.344 | 4.231 | 4.672 | 4.620 | 4.017 |
| 2026-10-14 | 2026-09-25..2026-10-13 | 0.783 | 2.269 | 2.850 | 3.256 | 2.639 | 0.979 |
| 2026-10-15 | 2026-09-25..2026-10-14 | 3.200 | 2.378 | 4.097 | 3.796 | 3.873 | 4.000 |
| 2026-10-16 | 2026-09-25..2026-10-15 | 3.060 | 1.581 | 3.506 | 3.247 | 3.295 | 3.825 |
| 2026-10-17 | 2026-09-25..2026-10-16 | 0.000 | 0.627 | 0.784 | 0.428 | 0.754 | 0.000 |
| 2026-10-18 | 2026-09-25..2026-10-17 | 1.046 | 0.640 | 1.757 | 1.029 | 1.624 | 1.307 |
| 2026-10-19 | 2026-09-25..2026-10-18 | 3.385 | 2.661 | 4.359 | 4.761 | 4.688 | 4.231 |
| 2026-10-20 | 2026-09-25..2026-10-19 | 3.106 | 2.485 | 3.433 | 4.832 | 3.432 | 3.883 |
| 2026-10-21 | 2026-09-25..2026-10-20 | 3.303 | 1.846 | 4.168 | 4.026 | 3.695 | 4.129 |
| 2026-10-22 | 2026-09-25..2026-10-21 | 4.096 | 2.546 | 5.364 | 5.748 | 5.530 | 5.120 |

</details>

_Updated 2026-10-24 00:58._
<!-- /forecast:synthetic data, starter rules off, focus=shifts<=2 -->

<!-- conditions:synthetic data, starter rules off, focus=shifts<=2 -->
## Focus conditions (synthetic data, starter rules off, focus=shifts<=2)

200 sessions (a gap of 15+ min starts a new one); a session is focused with >= 25 focus minutes (28% are). Holdout: train 2026-09-25..2026-10-08, test 2026-10-09..2026-10-22.

| method | ROC-AUC (higher) | Brier (lower) | log-loss (lower) | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|---|
| TabPFN | 0.804 | 0.180 | 0.535 | 96 | 104 | 107.177 |
| base rate | 0.500 | 0.228 | 0.651 | 96 | 104 | 0.000 |
| rate by start hour | 0.700 | 0.198 | 0.595 | 96 | 104 | 0.000 |

Observed rates, conditions seen at least 8 times:

| condition | sessions | focused | rate | 90% interval |
|---|---|---|---|---|
| started in the morning 06-12 | 40 | 24 | 60% | 47%-72% |
| started with Work | 63 | 36 | 57% | 47%-67% |
| previous session was mostly Work | 69 | 25 | 36% | 27%-46% |
| previous session was mostly Comms | 18 | 6 | 33% | 18%-53% |
| started in the afternoon 12-18 | 68 | 22 | 32% | 24%-42% |
| previous session was mostly Uncategorized | 43 | 11 | 26% | 16%-38% |
| previous session was mostly Media | 69 | 14 | 20% | 14%-29% |
| started with Uncategorized | 20 | 4 | 20% | 9%-38% |
| started with Comms | 32 | 6 | 19% | 10%-32% |
| started in the evening 18-24 | 79 | 11 | 14% | 9%-22% |
| started with Media | 85 | 11 | 13% | 8%-20% |
| started in the night 00-06 | 13 | 0 | 0% | 0%-17% |

Routine score 0.57 (routine mode). Suggestion: start in the morning 06-12 with Work instead of the evening 18-24 with Media (35% -> 85%, seen 20 times).

Leakage check: PASS: features are start-time context only.

_Updated 2026-10-24 01:45._
<!-- /conditions:synthetic data, starter rules off, focus=shifts<=2 -->

<!-- conditions:real data, starter rules on, focus=shifts<=2 -->
## Focus conditions (real data, starter rules on, focus=shifts<=2)

295 sessions (a gap of 15+ min starts a new one); a session is focused with >= 25 focus minutes (19% are). Holdout: train 2026-07-17..2026-10-09, test 2026-10-10..2026-10-23.

| method | ROC-AUC (higher) | Brier (lower) | log-loss (lower) | n_train | n_test | fit+predict s |
|---|---|---|---|---|---|---|
| TabPFN | 0.747 | 0.119 | 0.387 | 239 | 56 | 125.194 |
| base rate | 0.500 | 0.126 | 0.421 | 239 | 56 | 0.000 |
| rate by start hour | 0.516 | 0.127 | 0.425 | 239 | 56 | 0.000 |

Observed rates, conditions seen at least 8 times:

| condition | sessions | focused | rate | 90% interval |
|---|---|---|---|---|
| started with Work | 129 | 46 | 36% | 29%-43% |
| started with Comms | 9 | 2 | 22% | 8%-50% |
| previous session was mostly Work | 128 | 28 | 22% | 16%-28% |
| started in the afternoon 12-18 | 101 | 22 | 22% | 16%-29% |
| started in the night 00-06 | 70 | 14 | 20% | 13%-29% |
| previous session was mostly Uncategorized | 127 | 22 | 17% | 12%-24% |
| started in the evening 18-24 | 111 | 18 | 16% | 11%-23% |
| started in the morning 06-12 | 13 | 2 | 15% | 5%-37% |
| started with Media | 30 | 4 | 13% | 6%-27% |
| previous session was mostly Media | 35 | 4 | 11% | 5%-23% |
| started with Uncategorized | 127 | 4 | 3% | 1%-7% |

Routine score 0.33 (design mode). Suggestion: start in the evening 18-24 with Work instead of the evening 18-24 with Uncategorized (7% -> 44%, seen 39 times).

Leakage check: PASS: features are start-time context only.

_Updated 2026-10-24 01:41._
<!-- /conditions:real data, starter rules on, focus=shifts<=2 -->
