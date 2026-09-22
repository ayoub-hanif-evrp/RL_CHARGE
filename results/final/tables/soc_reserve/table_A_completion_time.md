# Table A — soc_reserve performance (failures retained; H for infeasible completion)

| method | n_routes | n_seeds | n_rows | route_weighted_feasibility | parent_balanced_feasibility | feasibility_ci95_lo | feasibility_ci95_hi | route_weighted_completion_all | parent_balanced_completion_all | completion_all_ci95_lo | completion_all_ci95_hi | seed_sd_completion_all | uncertainty | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GreedyMinimumSufficientCharge | 49 | 1 | 196 | 0.03571428571428571 | 0.010802469135802469 | 0.0 | 0.032407407407407406 | 1068.7285287654397 | 1175.1709747500404 | 481.6666666666667 | 2145.3419495000812 |  | parent_cluster | 0 |
| HybridPPO | 49 | 5 | 980 | 0.07755102040816327 | 0.05339506172839506 | 0.031249999999999997 | 0.08209876543209876 | 1057.5060921660347 | 1169.2331287777986 | 810.3462589900504 | 1574.6564146450755 | 3.4706503614590054 | hierarchical_seed_then_parent | 42,43,44,45,46 |
