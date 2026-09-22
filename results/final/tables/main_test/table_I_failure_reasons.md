# Table I — main_test failure reasons (failures retained in all-routes metrics)

| method | n_rows | n_feasible | n_failed | reason | count |
| --- | --- | --- | --- | --- | --- |
| AttentionPPO | 245 | 26 | 219 | NO_FEASIBLE_ACTION | 216 |
| AttentionPPO | 245 | 26 | 219 | ZERO_CHARGE_NOOP | 3 |
| AttentionPPO | 245 | 26 | 219 | feasible | 26 |
| DiscretePPO | 245 | 19 | 226 | NO_FEASIBLE_ACTION | 214 |
| DiscretePPO | 245 | 19 | 226 | ZERO_CHARGE_NOOP | 12 |
| DiscretePPO | 245 | 19 | 226 | feasible | 19 |
| GreedyFullCharge | 49 | 4 | 45 | NO_FEASIBLE_ACTION | 36 |
| GreedyFullCharge | 49 | 4 | 45 | ZERO_CHARGE_NOOP | 9 |
| GreedyFullCharge | 49 | 4 | 45 | feasible | 4 |
| GreedyMinimumSufficientCharge | 49 | 2 | 47 | NO_FEASIBLE_ACTION | 41 |
| GreedyMinimumSufficientCharge | 49 | 2 | 47 | ZERO_CHARGE_NOOP | 6 |
| GreedyMinimumSufficientCharge | 49 | 2 | 47 | feasible | 2 |
| HybridPPO | 245 | 25 | 220 | NO_FEASIBLE_ACTION | 218 |
| HybridPPO | 245 | 25 | 220 | ZERO_CHARGE_NOOP | 2 |
| HybridPPO | 245 | 25 | 220 | feasible | 25 |
| OneStepLookahead | 49 | 5 | 44 | NO_FEASIBLE_ACTION | 43 |
| OneStepLookahead | 49 | 5 | 44 | ZERO_CHARGE_NOOP | 1 |
| OneStepLookahead | 49 | 5 | 44 | feasible | 5 |
