# SynthCharge Table C — Feasible-only operational

Secondary metrics; failures dropped.

| method | feasible_only_completion | charging_time | station_visits | travel_time | waiting_time | terminal_soc |
|---|---|---|---|---|---|---|
| HybridPPO | 3.787 | 0.297 | 2.07 | 2.163 | 1.157 | 0.390 |
| DiscretePPO | 3.690 | 0.284 | 1.95 | 2.119 | 1.116 | 0.386 |
| GreedyMinimumSufficientCharge | 2.992 | 0.021 | 0.44 | NA | NA | NA |
| GreedyFullCharge | 3.337 | 0.205 | 0.56 | NA | NA | NA |
| OneStepLookahead | 3.415 | 0.239 | 2.82 | NA | NA | NA |
