# SynthCharge Table B — Charging-required

72 charging-required TEST routes.

| method | n_routes | feasibility_mean | feasibility_sd | completion_all_mean | charging_time | station_visits |
|---|---|---|---|---|---|---|
| HybridPPO | 72 | 0.881 | 0.032 | 4.770 | 0.356 | 2.45 |
| DiscretePPO | 72 | 0.883 | 0.019 | 4.644 | 0.338 | 2.28 |
| GreedyMinimumSufficientCharge | 72 | 0.194 | NA | 8.704 | 0.048 | 1.00 |
| GreedyFullCharge | 72 | 0.319 | NA | 8.025 | 0.366 | 1.00 |
| OneStepLookahead | 72 | 0.583 | NA | 6.318 | 0.296 | 3.43 |
