# FA-HPPO failure analysis (frozen V3 raw)

**No TEST replay. No unrecorded trajectory reconstruction.**

- Routes: 180
- Route×seed failures: 52
- Routes with any failure: 20
- Routes failing all 5 seeds: 5
- Routes failing exactly 1 seed: 9
- Failures per seed: {'42': 9, '43': 10, '44': 9, '45': 8, '46': 16}
- Seed 46 vs 45: Seed 46 has 16 route failures vs 8 for seed 45.
- Reasons: {'NO_FEASIBLE_ACTION': 52}

## Answers

- **few_hard_routes:** 5 routes fail for all five seeds; 20 routes have ≥1 seed failure out of 180.
- **same_routes_all_seeds:** 5 routes fail for every seed (see all_seeds_fail_routes).
- **layout_length:** See tableA03 / per-route records; failures are charging_required only in frozen rows inspected.
- **seed_46_vs_45:** Seed 46 has 16 route failures vs 8 for seed 45.

## Unavailable without TEST replay

- Pre-failure action/SOC trajectories were not stored in V3 raw rows.
- Causal step-level diagnosis would require forbidden TEST replay.

Publication table: `results_paper/tables/tableA03_failure_routes.*`
Figure: `results_paper/figures/appendix/figA05_failure_consistency.png`
