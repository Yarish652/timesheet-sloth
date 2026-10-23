# Evaluation

Numbers written by `sloth eval`. Each section is replaced when its command is re-run.
Real-data sections are computed locally and contain aggregates only.

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

<!-- forecast:real data, starter rules on -->
## Focus forecast backtest (real data, starter rules on)

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
<!-- /forecast:real data, starter rules on -->

<!-- forecast:synthetic data, starter rules off -->
## Focus forecast backtest (synthetic data, starter rules off)

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
<!-- /forecast:synthetic data, starter rules off -->
