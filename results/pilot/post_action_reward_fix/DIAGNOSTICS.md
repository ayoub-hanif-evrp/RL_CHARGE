# Post-fix deterministic TRAIN/VAL diagnostics

TEST was not used. The three methods are **not** an exact feasibility oracle.
Their union is only a diagnostic lower bound on “easy” routes.

Station revisits are charges that reselect a station already used since the
last frozen customer. After the segment-revisit mask, legal baseline
behavior has **zero** revisits and **zero** loop-guard hits.

## TRAIN (132 routes)

| Method | Feasible | Rate | Parent-balanced feas. | Mean station visits | Revisits | Loop-guard | ZERO_CHARGE_NOOP | NO_FEASIBLE_ACTION |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreedyMinimumSufficientCharge | 3 / 132 | 2.3% | 2.1% | 2.30 | 0 | 0 | 22 | 107 |
| GreedyFullCharge | 8 / 132 | 6.1% | 7.5% | 4.20 | 0 | 0 | 33 | 91 |
| OneStepLookahead | 9 / 132 | 6.8% | 6.1% | 6.00 | 0 | 0 | 0 | 123 |

Union (diagnostic only): **14 / 132 = 10.6%**.

## VALIDATION (77 routes)

| Method | Feasible | Rate | Parent-balanced feas. | Mean station visits | Revisits | Loop-guard | ZERO_CHARGE_NOOP | NO_FEASIBLE_ACTION |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GreedyMinimumSufficientCharge | 21 / 77 | 27.3% | 9.7% | 1.79 | 0 | 0 | 9 | 47 |
| GreedyFullCharge | 18 / 77 | 23.4% | 8.5% | 2.74 | 0 | 0 | 23 | 36 |
| OneStepLookahead | 12 / 77 | 15.6% | 5.3% | 5.27 | 0 | 0 | 0 | 65 |

Union (diagnostic only): **23 / 77 = 29.9%**.

Route-weighted VAL still looks easier than TRAIN because parent `r102` is
41 of 77 VAL routes. Parent-balanced VAL feasibility is ~9.7% even for the
best of these three heuristics.

Greedy `ZERO_CHARGE_NOOP` remains a first-visit backstop (target SOC equal
to arrival SOC), not a same-station reselection loop.
