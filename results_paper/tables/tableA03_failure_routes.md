# Table A3 — FA-HPPO failing routes (frozen V3 raw)

From frozen raw rows only. Alias R01–R20 matches Fig. A6 order (by n_seeds_fail, then route_id). Pre-failure trajectories were not recorded and are unavailable without TEST replay (forbidden). Shared failures = n_seeds_fail=5.

| Alias | Route | Layout | Length | Charge class | n_seeds_fail | Fail seeds | OK seeds | Reason | Mean visits (fail) | Mean terminal SOC (fail) |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | v3sc_sc_C_n30_s400121_01 | C | long | charging_required | 5 | 42,43,44,45,46 | — | NO_FEASIBLE_ACTION | 0.00 | 0.111 |
| R02 | v3sc_sc_RC_n15_s400281_00 | RC | short | charging_required | 5 | 42,43,44,45,46 | — | NO_FEASIBLE_ACTION | 0.00 | 0.180 |
| R03 | v3sc_sc_R_n15_s400099_00 | R | medium | charging_required | 5 | 42,43,44,45,46 | — | NO_FEASIBLE_ACTION | 0.00 | 0.055 |
| R04 | v3sc_sc_R_n15_s400108_01 | R | short | charging_required | 5 | 42,43,44,45,46 | — | NO_FEASIBLE_ACTION | 0.00 | 0.214 |
| R05 | v3sc_sc_R_n15_s400369_02 | R | short | charging_required | 5 | 42,43,44,45,46 | — | NO_FEASIBLE_ACTION | 0.20 | 0.188 |
| R06 | v3sc_sc_RC_n50_s400161_01 | RC | long | charging_required | 4 | 42,43,44,46 | 45 | NO_FEASIBLE_ACTION | 5.00 | 0.086 |
| R07 | v3sc_sc_R_n30_s400093_00 | R | medium | charging_required | 4 | 42,43,44,46 | 45 | NO_FEASIBLE_ACTION | 0.00 | 0.027 |
| R08 | v3sc_sc_RC_n15_s400092_00 | RC | medium | charging_required | 3 | 42,43,46 | 44,45 | NO_FEASIBLE_ACTION | 3.67 | 0.172 |
| R09 | v3sc_sc_RC_n15_s400128_02 | RC | short | charging_required | 3 | 43,44,46 | 42,45 | NO_FEASIBLE_ACTION | 0.33 | 0.089 |
| R10 | v3sc_sc_C_n15_s402710_00 | C | short | charging_required | 2 | 45,46 | 42,43,44 | NO_FEASIBLE_ACTION | 1.50 | 0.276 |
| R11 | v3sc_sc_RC_n15_s400497_02 | RC | short | charging_required | 2 | 42,45 | 43,44,46 | NO_FEASIBLE_ACTION | 1.00 | 0.260 |
| R12 | v3sc_sc_C_n15_s400460_01 | C | short | charging_required | 1 | 43 | 42,44,45,46 | NO_FEASIBLE_ACTION | 1.00 | 0.270 |
| R13 | v3sc_sc_C_n15_s402638_02 | C | short | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 0.00 | 0.201 |
| R14 | v3sc_sc_RC_n15_s400317_02 | RC | short | charging_required | 1 | 45 | 42,43,44,46 | NO_FEASIBLE_ACTION | 0.00 | 0.017 |
| R15 | v3sc_sc_RC_n15_s400407_01 | RC | short | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 0.00 | 0.056 |
| R16 | v3sc_sc_RC_n30_s400005_01 | RC | long | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 1.00 | 0.088 |
| R17 | v3sc_sc_R_n15_s400072_00 | R | short | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 0.00 | 0.100 |
| R18 | v3sc_sc_R_n15_s400306_02 | R | short | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 1.00 | 0.202 |
| R19 | v3sc_sc_R_n30_s400120_00 | R | medium | charging_required | 1 | 44 | 42,43,45,46 | NO_FEASIBLE_ACTION | 2.00 | 0.178 |
| R20 | v3sc_sc_R_n50_s400114_02 | R | long | charging_required | 1 | 46 | 42,43,44,45 | NO_FEASIBLE_ACTION | 0.00 | 0.033 |
