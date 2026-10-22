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
