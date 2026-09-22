# Final experiment report

Generated from tracked raw records after analysis-only cleanup. No retraining. TEST was not reopened for tuning.

## A. Frozen methodology and provenance

- paper_code_sha: `a175ee43548a5d2d5154a9ae0731a01f7642a7f6`
- results_analysis_code_sha: `7735be30a874517d26b977051159102a3c1190b9`
- cleanup_commit_sha: the git commit that contains this cleanup; newer than `results_analysis_code_sha`; not written into this file.
- Device: CPU. DDQN excluded. Methodology frozen. TEST is consumed.

## B–P. See generated tables

Main TEST population remains 49 routes × declared methods. Failures retained. H used for infeasible completion.
Matched terrain sibling groups on TEST: **11** complete L/NL/VG groups. Unmatched NL-only Medium/Large routes are excluded from terrain-effect claims.

### HybridPPO seed 46

Seed 46 exhibited late-training collapse (validation feasibility reached 0 at updates 220–260). The selected checkpoint is update 60, which had nonzero validation feasibility. Predeclared validation checkpoint selection prevented the collapsed final policy from being used for TEST. Seed 46 was not retrained.

### Per-seed HybridPPO TEST feasibility (49 routes each; not a pooled unique-route count)

- seed 42: 5/49 route-weighted feas. 0.1020; parent-balanced 0.0787
- seed 43: 6/49 route-weighted feas. 0.1224; parent-balanced 0.0849
- seed 44: 6/49 route-weighted feas. 0.1224; parent-balanced 0.0849
- seed 45: 3/49 route-weighted feas. 0.0612; parent-balanced 0.0664
- seed 46: 5/49 route-weighted feas. 0.1020; parent-balanced 0.0926

Mean feasible routes per seed: 5.00/49 (SD 1.22). Pooled 25/245 is 25 seed-route episodes, not 25 unique TEST routes.

### Known-feasible small-route diagnostic (not model selection)

17 / 34 eligible ≤5-customer TEST routes have a proven feasible solution under the restricted {continuation-minimum, full} action set. This is not globally exact for continuous Hybrid PPO.

- GreedyMinimumSufficientCharge on those 17 routes: route-weighted feas. 0.058823529411764705; mean feasible routes/seed 1.0.
- GreedyFullCharge on those 17 routes: route-weighted feas. 0.17647058823529413; mean feasible routes/seed 3.0.
- OneStepLookahead on those 17 routes: route-weighted feas. 0.29411764705882354; mean feasible routes/seed 5.0.
- HybridPPO on those 17 routes: route-weighted feas. 0.15294117647058825; mean feasible routes/seed 2.6.
- DiscretePPO on those 17 routes: route-weighted feas. 0.11764705882352941; mean feasible routes/seed 2.0.
- AttentionPPO on those 17 routes: route-weighted feas. 0.16470588235294117; mean feasible routes/seed 2.8.

A miss on a restricted-proven-feasible route is **policy failure**. Timeout or restricted-infeasible rows do **not** establish continuous-action infeasibility.

### Nonlinear sensitivity

The Montoya/native nonlinear experiment is **excluded_from_paper** / `invalid_external_sensitivity`. It is not a completed scientific sensitivity analysis.

Large_Network HybridPPO TEST feasibility: 0.0.

## Q. Limitations that must appear in the manuscript

- HybridPPO does not show a statistically significant advantage over the declared comparators.
- All Holm-adjusted main-comparison p-values are non-significant (all Holm p = 1.0).
- Feasibility is low across all methods.
- All learned methods have 0% feasibility on Large_Network TEST routes.
- Some restricted-search-proven feasible small routes are still missed by HybridPPO.
- Ablations do not show clear support for every architectural component.
- A5/pooling is weaker descriptively, but evidence is limited.
- TEST has only six parent clusters.
- Do not describe HybridPPO as best, superior, or state of the art.

## R. Manuscript-ready tables

- `results/final/tables/main_test/table_A_completion_time.md`
- `results/final/tables/main_test/table_A_per_seed_feasibility.md`
- `results/final/tables/main_test/table_B_feasible_only.md`
- `results/final/tables/main_test/table_C_terrain.md`
- `results/final/tables/main_test/table_D_size_family.md`
- `results/final/tables/ablation/table_E_ablations.md`
- `results/final/tables/frvcpy_native/table_F_frvcpy.md`
- `results/final/tables/main_test/table_G_runtime.md`
- `results/final/tables/soc_reserve/table_H_soc_reserve.md`
- `results/final/tables/main_test/table_I_failure_reasons.md`
- `results/final/statistics/main_test/paired_holm.csv`
- `results/final/statistics/exact_small/known_feasible_diagnostic.json`

## S. Manuscript-ready figures

- `results/final/figures/main_test/hybrid_ppo_val_curves.png`
- `results/final/figures/main_test/feasibility_bars.png`
- `results/final/figures/main_test/completion_all_bars.png`
- `results/final/figures/main_test/terrain_comparison.png`
- `results/final/figures/main_test/hybrid_failure_reasons.png`
- `results/final/figures/ablation/ablation_comparison.png`
- `results/final/figures/soc_reserve/soc_reserve.png`
- `results/final/figures/frvcpy_native/frvcp_gap_hist.png`

Nonlinear figures are excluded.

## T. Warnings

1. No post-TEST tuning is allowed. TEST is consumed.
2. Restricted search is not globally exact.
3. frvcpy is not exact for EVRPTW-GR.
4. SOC reserve is not part of main_test sample sizes.
5. Nonlinear Montoya output is invalid and excluded.
6. Terrain claims use matched L/NL/VG siblings only.
7. Parent-balanced CIs must not be labeled as route-weighted.
8. Legacy DDQN is not a paper method.

