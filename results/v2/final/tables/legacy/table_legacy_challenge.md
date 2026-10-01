# Legacy same-domain challenge (NOT a fresh TEST)

26 official skeletons of six historically consumed V1 TEST parents. Gold-trained models only.

| method | n_routes | n_seeds | mean_feasible_routes_per_seed | feasibility_mean | feasibility_sd | feasibility_ci95_lo | feasibility_ci95_hi | completion_all_mean | completion_all_sd | completion_all_ci95_lo | completion_all_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HybridPPO | 26 | 5 | 21.20 | 0.815 | 0.050 | 0.756 | 0.942 | 902.808 | 15.277 | 534.538 | 995.206 |
| DiscretePPO | 26 | 5 | 23.80 | 0.915 | 0.050 | 0.822 | 0.989 | 888.999 | 11.323 | 525.551 | 988.274 |
| GreedyMinimumSufficientCharge | 26 | 1 | 13.00 | 0.500 | NA | NA | NA | 999.683 | NA | NA | NA |
| GreedyFullCharge | 26 | 1 | 17.00 | 0.654 | NA | NA | NA | 973.162 | NA | NA | NA |
| OneStepLookahead | 26 | 1 | 19.00 | 0.731 | NA | NA | NA | 963.407 | NA | NA | NA |
