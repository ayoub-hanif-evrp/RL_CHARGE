# Table 1 — V4 TEST main results

Fresh independently generated held-out SynthCharge TEST (180 routes).
FA-HPPO intervals are 95% Student-t confidence intervals across five independently trained seeds.
Deterministic baselines have no uncertainty estimates.

| Method | Feasibility | Failure-retaining completion | Runtime (s) |
|---|---:|---:|---:|
| FA-HPPO | 94.56% [92.30, 96.81] | 4.140 [4.081, 4.199] | 0.1015 |
| One-step lookahead | 68.89% | 5.452 | 0.1067 |
| Greedy full charge | 52.78% | 6.580 | 0.0405 |
| Greedy minimum charge | 44.44% | 6.970 | 0.0408 |
