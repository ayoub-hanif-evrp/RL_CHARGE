# Table 2 — Main V3 TEST results

FA-HPPO = mean over 5 training seeds; baselines deterministic. Infeasible completion = horizon H. CI = hierarchical bootstrap (seed→route). All Holm-adjusted primary p < 0.001.

| Method | All feas. | 95% CI | All completion | Charge-req feas. | Charge-req completion | Runtime (s) |
|---|---|---|---|---|---|---|
| FA-HPPO | 0.942 | [0.920, 0.961] | 4.200 | 0.928 | 4.456 | 0.130 |
| Lookahead | 0.700 | NA | 5.349 | 0.625 | 5.935 | 0.151 |
| Greedy Full | 0.467 | NA | 6.904 | 0.333 | 7.891 | 0.055 |
| Greedy Min | 0.394 | NA | 7.271 | 0.243 | 8.350 | 0.053 |
