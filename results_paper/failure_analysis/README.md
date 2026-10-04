# FA-HPPO failure analysis (frozen V3 raw)

**No TEST replay. No unrecorded trajectory reconstruction.**

- Routes: 180
- Route×seed failures: 52
- Routes with any failure: 20
- Routes failing all 5 seeds: 5
- Routes failing exactly 1 seed: 9
- Failures per seed: {'42': 9, '43': 10, '44': 9, '45': 8, '46': 16}
- Seed 46 vs 45: seed45=8 failures; seed46=16 failures.
- Reasons: {'NO_FEASIBLE_ACTION': 52}

## Answers

- **few_hard_routes:** 20/180 routes fail for >=1 seed; 160/180 succeed for all 5 seeds; 5 fail for all 5.
- **same_routes_all_seeds:** 5 routes fail under all five seeds: v3sc_sc_C_n30_s400121_01, v3sc_sc_RC_n15_s400281_00, v3sc_sc_R_n15_s400099_00, v3sc_sc_R_n15_s400108_01, v3sc_sc_R_n15_s400369_02
- **layout_length_concentration:** See figA05 panel (b) / tableA03; computed from frozen raw only.
- **seed_46_vs_45:** seed45=8 failures; seed46=16 failures.
- **all_ok_vs_multi_fail:** 160 routes all-seed OK vs 20 with any failure; distinguishing features beyond layout/length/charge_class require trajectories not stored in V3 raw.

## Unavailable without TEST replay

- step-level pre-failure TEST trajectories (not recorded in V3 raw)
- causal action sequences immediately before NO_FEASIBLE_ACTION without TEST replay

Publication table: `results_paper/tables/tableA03_failure_routes.*`
Figure (official builder): `results_paper/figures/appendix/figA07_failure_consistency.png , figA08_failure_heatmap.png`

Regenerate publication figures with `python scripts/paper/build_results_paper.py`.
