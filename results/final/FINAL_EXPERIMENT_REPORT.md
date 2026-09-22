# Final experiment report

This report is generated from tracked raw records. It does not retune models.

## A. Frozen methodology and provenance

- PAPER_CODE_SHA (training/evaluation science): `a175ee43548a5d2d5154a9ae0731a01f7642a7f6`
- Analysis-code SHA (this results commit parent / working tree): `887c97f34177a422bc0188c2ca08c07f1989a028`
- Device used for training and evaluation: CPU (`torch` CPU)
- Frozen population: 258 routes; TRAIN 132 / VALIDATION 77 / TEST 49; parents 20 / 6 / 6.
- Proposed method only: Hybrid PPO (WHEN + WHERE + HOW MUCH).
- Learned comparators: DiscretePPO, AttentionPPO. Deterministic: GreedyMinimumSufficientCharge, GreedyFullCharge, OneStepLookahead.
- Ablations: FULL=HybridPPO, A1=DiscretePPO, A2–A5 dedicated checkpoints. Seeds 42–44.
- Legacy DDQN is not trained, not evaluated, and not tabulated.
- TEST was opened only after checkpoint freeze. No model or hyperparameter change after freeze.

Pinned hashes:

- `corpus.jsonl`: `1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0`
- `manifest.csv`: `41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d`
- `corpus_metadata.json`: `1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96`
- `train.json`: `3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce`
- `validation.json`: `1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4`
- `test.json`: `9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0`
- `split_metadata.json`: `4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900`

## B. Exact final command sequence

All TRAIN/VAL/TEST science commands ran in worktree `RL_CHARGE_PAPER` at `PAPER_CODE_SHA` with a clean tree.

```
python scripts/train_rl.py --method hybrid_ppo --split train --seeds {42..46}
python scripts/train_rl.py --method discrete_ppo --split train --seeds {42..46}
python scripts/train_rl.py --method attention_ppo --split train --seeds {42..46}
python scripts/run_ablations.py --seeds {42,43,44} --variants A2,A3,A4,A5
python scripts/run_baselines.py --split test --scenario main_test --run-id final_main_test_baselines
python scripts/evaluate.py --split test --scenario main_test --methods hybrid_ppo,discrete_ppo,attention_ppo --seeds paper --run-id final_main_test_learned
python scripts/evaluate.py --split test --scenario ablation --methods hybrid_ppo --seeds ablation --run-id final_ablation_FULL
python scripts/evaluate.py --split test --scenario ablation --methods discrete_ppo --seeds ablation --run-id final_ablation_A1
python scripts/evaluate.py --split test --scenario ablation --methods A2,A3,A4,A5 --seeds ablation --run-id final_ablation_A2_A5
python scripts/run_soc_reserve.py --split test --scenario soc_reserve --levels 0,0.05,0.10,0.15 --seed {42..46} --run-id final_soc_reserve_seed{seed}
python scripts/run_exact_small.py --split test --max-customers 5 --max-expansions 20000 --scenario exact_small --run-id final_exact_small_test
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --parity-only
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --scenario frvcpy_native --run-id final_frvcpy_native
python scripts/run_frvcpy_benchmark.py --data data/external/frvcpy --montoya --scenario nonlinear_sensitivity --run-id final_montoya_nonlinear
```

Analysis (this tree, after copying raw JSONL):

```
python scripts/analyze_results.py --raw results/final/raw --scenario <scenario> --out results/final/statistics/<scenario>
python scripts/make_tables.py --raw results/final/raw --scenario <scenario> --out results/final/tables
python scripts/make_figures.py --raw results/final/raw --scenario <scenario> --out results/final/figures --curves-root results/final/training/HybridPPO
```

## C. Training summary for every learned seed

| method | seed | best update | parent-balanced VAL feasibility | route-weighted VAL feasibility | parent-balanced VAL completion | status | runtime_s | finite losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HybridPPO | 42 | 200 | 0.096980 | 0.272727 | 1170.4712 | early_stop | 4866.2 | True |
| HybridPPO | 43 | 280 | 0.101045 | 0.285714 | 1170.3969 | completed | 5203.0 | True |
| HybridPPO | 44 | 220 | 0.096980 | 0.272727 | 1170.5353 | completed | 5313.3 | True |
| HybridPPO | 45 | 350 | 0.088850 | 0.246753 | 1170.9756 | completed | 5186.1 | True |
| HybridPPO | 46 | 60 | 0.064654 | 0.181818 | 1172.4407 | early_stop | 3071.9 | True |
| DiscretePPO | 42 | 150 | 0.076655 | 0.207792 | 1171.7498 | early_stop | 3839.5 | True |
| DiscretePPO | 43 | 190 | 0.088850 | 0.246753 | 1170.9166 | early_stop | 4436.2 | True |
| DiscretePPO | 44 | 50 | 0.064170 | 0.116883 | 1167.1008 | early_stop | 2812.5 | True |
| DiscretePPO | 45 | 200 | 0.108498 | 0.233766 | 1166.3623 | early_stop | 4429.6 | True |
| DiscretePPO | 46 | 390 | 0.064654 | 0.181818 | 1172.0959 | completed | 3702.2 | True |
| AttentionPPO | 42 | 70 | 0.092915 | 0.259740 | 1170.6774 | early_stop | 2742.6 | True |
| AttentionPPO | 43 | 360 | 0.101045 | 0.285714 | 1170.5174 | completed | 4162.0 | True |
| AttentionPPO | 44 | 90 | 0.092915 | 0.259740 | 1170.8491 | early_stop | 3018.9 | True |
| AttentionPPO | 45 | 150 | 0.104433 | 0.220779 | 1165.6843 | early_stop | 3602.9 | True |
| AttentionPPO | 46 | 280 | 0.096980 | 0.272727 | 1170.5708 | completed | 4099.9 | True |
| ablation_A2 | 42 | 310 | 0.096690 | 0.220779 | 1161.4666 | completed | 3157.8 | True |
| ablation_A2 | 43 | 380 | 0.048587 | 0.142857 | 1172.3478 | completed | 3205.4 | True |
| ablation_A2 | 44 | 40 | 0.068912 | 0.207792 | 1171.7196 | early_stop | 1936.6 | True |
| ablation_A3 | 42 | 40 | 0.116628 | 0.259740 | 1161.2293 | early_stop | 2385.6 | True |
| ablation_A3 | 43 | 140 | 0.096980 | 0.272727 | 1170.5172 | early_stop | 3320.7 | True |
| ablation_A3 | 44 | 140 | 0.096980 | 0.272727 | 1170.5215 | early_stop | 3346.9 | True |
| ablation_A4 | 42 | 330 | 0.096980 | 0.272727 | 1170.7746 | completed | 3958.3 | True |
| ablation_A4 | 43 | 200 | 0.096980 | 0.272727 | 1170.5148 | early_stop | 4100.5 | True |
| ablation_A4 | 44 | 180 | 0.101045 | 0.285714 | 1170.7685 | early_stop | 3972.5 | True |
| ablation_A5 | 42 | 190 | 0.096980 | 0.272727 | 1170.7011 | early_stop | 3848.9 | True |
| ablation_A5 | 43 | 130 | 0.096980 | 0.272727 | 1170.6094 | early_stop | 3174.1 | True |
| ablation_A5 | 44 | 20 | 0.096980 | 0.272727 | 1170.5075 | early_stop | 2073.3 | True |

Sum of recorded training runtimes: 98967.6 s (27.49 h).
Every training manifest reports `git_sha=PAPER_CODE_SHA` and `git_dirty=false`.
Normalizers were fit on TRAIN only (132 routes). Checkpoint selection used all 77 VALIDATION routes.

## D. Main TEST result summary

Population: 49 TEST routes, 6 parents. Failures retained. All-routes completion uses H for infeasible episodes.

| method | n_rows | n_seeds | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) | parent-balanced completion (H) | seed SD completion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GreedyMinimumSufficientCharge | 49 | 1 | 0.040816 | 0.012346 | 1067.9322 | 1174.9301 | — |
| GreedyFullCharge | 49 | 1 | 0.081633 | 0.109568 | 1054.4000 | 1142.1687 | — |
| OneStepLookahead | 49 | 1 | 0.102041 | 0.030864 | 1045.3440 | 1168.0979 | — |
| HybridPPO | 245 | 5 | 0.102041 | 0.081481 | 1056.7494 | 1164.3678 | 6.0658 |
| DiscretePPO | 245 | 5 | 0.077551 | 0.084259 | 1055.2661 | 1154.3223 | 10.1441 |
| AttentionPPO | 245 | 5 | 0.106122 | 0.079938 | 1058.3349 | 1169.9384 | 3.7180 |

Main TEST row-count audit: 882 rows; methods {'GreedyMinimumSufficientCharge': 49, 'GreedyFullCharge': 49, 'OneStepLookahead': 49, 'HybridPPO': 245, 'DiscretePPO': 245, 'AttentionPPO': 245}.

## E. Deterministic baseline summary

- GreedyMinimumSufficientCharge: route-weighted feasibility 0.040816; parent-balanced feasibility 0.012346; all-routes completion (route-weighted) 1067.9322; feasible 2/49.
- GreedyFullCharge: route-weighted feasibility 0.081633; parent-balanced feasibility 0.109568; all-routes completion (route-weighted) 1054.4000; feasible 4/49.
- OneStepLookahead: route-weighted feasibility 0.102041; parent-balanced feasibility 0.030864; all-routes completion (route-weighted) 1045.3440; feasible 5/49.

## F. DiscretePPO comparison

- DiscretePPO route-weighted feasibility 0.077551 vs HybridPPO 0.102041.
- DiscretePPO parent-balanced feasibility 0.084259 vs HybridPPO 0.081481.
- DiscretePPO all-routes completion (route-weighted) 1055.2661 vs HybridPPO 1056.7494.
- Formal paired tests are in `results/final/statistics/main_test/paired_holm.csv` (parent-matched; learned side averaged over seeds; exact 2^6 sign-flip; Holm over the declared family).

## G. AttentionPPO comparison

- AttentionPPO route-weighted feasibility 0.106122 vs HybridPPO 0.102041.
- AttentionPPO parent-balanced feasibility 0.079938 vs HybridPPO 0.081481.
- AttentionPPO all-routes completion (route-weighted) 1058.3349 vs HybridPPO 1056.7494.

## H. Ablation summary

FULL reuses HybridPPO seeds 42–44. A1 reuses DiscretePPO seeds 42–44. A2–A5 are dedicated.

| variant | n_rows | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) | parent-balanced completion (H) |
| --- | --- | --- | --- | --- | --- |
| FULL | 147 | 0.115646 | 0.082819 | 1058.1689 | 1169.5939 |
| A1 | 147 | 0.081633 | 0.089506 | 1053.5772 | 1152.2650 |
| A2 | 147 | 0.102041 | 0.095679 | 1050.0463 | 1152.2562 |
| A3 | 147 | 0.108844 | 0.080761 | 1058.0825 | 1169.6534 |
| A4 | 147 | 0.122449 | 0.084877 | 1056.5298 | 1169.5834 |
| A5 | 147 | 0.081633 | 0.056584 | 1059.4012 | 1170.7864 |

Paired parent differences vs FULL (FULL minus variant; seeds averaged):

- A1: completion mean paired diff 17.328963749702; feasibility mean paired diff -0.006687242798353907; p_completion=1.0; p_feasibility=1.0; n_parents=6.
- A2: completion mean paired diff 17.337704120375108; feasibility mean paired diff -0.012860082304526746; p_completion=0.625; p_feasibility=1.0; n_parents=6.
- A3: completion mean paired diff -0.05940996257208061; feasibility mean paired diff 0.0020576131687242796; p_completion=0.75; p_feasibility=1.0; n_parents=6.
- A4: completion mean paired diff 0.010523479312349574; feasibility mean paired diff -0.0020576131687242796; p_completion=1.0; p_feasibility=1.0; n_parents=6.
- A5: completion mean paired diff -1.1924338643002699; feasibility mean paired diff 0.02623456790123457; p_completion=0.25; p_feasibility=0.25; n_parents=6.

## I. Terrain / size / family analysis

See `results/final/tables/main_test/table_C_terrain.md` and `table_D_size_family.md`.
Terrain uses the frozen sibling design (L / NL / VG). Size uses `network_group`. Family uses `customer_distribution`.

## J. SOC reserve sensitivity

Not used for model selection. GreedyMinimumSufficientCharge is deduplicated by `(route_id, min_soc_fraction)`.

| method | min_soc_fraction | n_rows | route-weighted feas. | parent-balanced feas. | route-weighted completion (H) |
| --- | --- | --- | --- | --- | --- |
| GreedyMinimumSufficientCharge | 0.0 | 49 | 0.040816 | 0.012346 | 1067.9322 |
| GreedyMinimumSufficientCharge | 0.05 | 49 | 0.040816 | 0.012346 | 1068.2503 |
| GreedyMinimumSufficientCharge | 0.1 | 49 | 0.040816 | 0.012346 | 1068.4643 |
| GreedyMinimumSufficientCharge | 0.15 | 49 | 0.020408 | 0.006173 | 1070.2673 |
| HybridPPO | 0.0 | 245 | 0.102041 | 0.081481 | 1056.7494 |
| HybridPPO | 0.05 | 245 | 0.114286 | 0.075309 | 1056.8690 |
| HybridPPO | 0.1 | 245 | 0.040816 | 0.012346 | 1057.7145 |
| HybridPPO | 0.15 | 245 | 0.053061 | 0.044444 | 1058.6914 |

## K. Restricted small-route reference

This is a restricted label-setting reference, not a global exact solver for continuous Hybrid PPO.
- Eligible TEST routes (max 5 customers): 34
- Status counts: {'timeout': 7, 'optimal_for_action_set': 17, 'infeasible': 10}
- Action set: CONTINUE + {continuation-minimum SOC, maximum SOC}
- `exact_for`: restricted_continuation_or_full_soc
- Do not call a HybridPPO difference an optimality gap for the continuous problem.
- HybridPPO on the same eligible routes: route-weighted feasibility 0.14705882352941177.

## L. Native frvcpy reference

Official upstream e-VRO/frvcpy reference routes/objectives. Exact only for the native compatible FRVCP formulation. Not EVRPTW-GR.

- frvcpy_Solver: n=133, feasible=133, mean gap vs frvcpy Solver (where defined)=None (n_gap=0).
- FRVCPGreedyMin: n=133, feasible=23, mean gap vs frvcpy Solver (where defined)=2.3052524293682946 (n_gap=23).
- FRVCPGreedyFull: n=133, feasible=29, mean gap vs frvcpy Solver (where defined)=16.16595670502378 (n_gap=29).

Do not report a HybridPPO-versus-frvcpy optimality gap on EVRPTW-GR.

## M. Nonlinear native sensitivity

Native Montoya/FRVCP nonlinear charging sensitivity. Not an original EVRPTW-GR result.

- frvcpy_Solver: n=22, feasible=22, mean gap vs frvcpy Solver (where defined)=None (n_gap=0).
- FRVCPGreedyMin: n=22, feasible=0, mean gap vs frvcpy Solver (where defined)=None (n_gap=0).
- FRVCPGreedyFull: n=22, feasible=0, mean gap vs frvcpy Solver (where defined)=None (n_gap=0).
- Several native nonlinear solver records store non-finite durations; those values are retained in raw JSONL and excluded from finite-duration means. This is not EVRPTW-GR.

## N. Statistical tests

Primary paired unit: `base_instance` (n=6 TEST parents).
Learned comparators: average training seeds within parent, then parent-matched differences.
p-values: exact two-sided sign-flip over all 2^6 assignments.
Holm correction: one family covering HybridPPO vs the five declared comparators on feasibility and all-routes completion.
Do not overstate significance with only six parents.
See `results/final/statistics/main_test/paired_holm.csv` and `parent_paired_differences.csv`.

## O. Failure reasons

Failures are retained in all-routes metrics. Histograms: `results/final/tables/main_test/table_I_failure_reasons.md`.
HybridPPO TEST reason counts: {'NO_FEASIBLE_ACTION': 218, 'feasible': 25, 'ZERO_CHARGE_NOOP': 2}.

## P. Runtime

- Training runtime sum (27 jobs): 98967.6 s.
- HybridPPO mean TEST episode runtime_s: 0.09368315347023688.
See `results/final/tables/*/table_G_runtime.md`.

## Q. Limitations

- TEST has six parent clusters; exact sign-flip p-values have coarse granularity (minimum two-sided p=1/32=0.03125 for a unanimous sign pattern).
- Restricted label-setting is not exact for continuous Hybrid PPO; many eligible TEST rows timed out at 20,000 expansions.
- frvcpy is a native FRVCP reference, not an EVRPTW-GR exact solver.
- Nonlinear Montoya results are external native-FRVCP sensitivity.
- All-routes completion assigns depot due date H to failures; feasible-only tables are conditional.
- Training and evaluation used CPU.

## R. Manuscript-ready tables

- `results/final/tables/main_test/table_A_completion_time.md`
- `results/final/tables/main_test/table_B_feasible_only.md`
- `results/final/tables/main_test/table_C_terrain.md`
- `results/final/tables/main_test/table_D_size_family.md`
- `results/final/tables/ablation/table_E_ablations.md`
- `results/final/tables/frvcpy_native/table_F_frvcpy.md`
- `results/final/tables/main_test/table_G_runtime.md`
- `results/final/tables/soc_reserve/table_H_soc_reserve.md`
- `results/final/tables/main_test/table_I_failure_reasons.md`
- `results/final/statistics/main_test/paired_holm.csv`
- `results/final/statistics/main_test/per_seed_*.json`

## S. Manuscript-ready figures

- `results/final/figures/main_test/hybrid_ppo_val_curves.png`
- `results/final/figures/main_test/feasibility_bars.png`
- `results/final/figures/main_test/completion_all_bars.png`
- `results/final/figures/main_test/terrain_comparison.png`
- `results/final/figures/main_test/hybrid_failure_reasons.png`
- `results/final/figures/ablation/ablation_comparison.png`
- `results/final/figures/soc_reserve/soc_reserve.png`
- `results/final/figures/frvcpy_native/frvcp_gap_hist.png`

## T. Warnings that must be stated in the paper

1. Do not call Hybrid PPO 'best' from a single metric; report both feasibility and all-routes completion, route-weighted and parent-balanced.
2. Do not treat restricted label-setting as globally exact, and do not call HybridPPO minus restricted-reference an optimality gap for the continuous problem.
3. Do not claim frvcpy is exact for EVRPTW-GR or report a HybridPPO–frvcpy EVRPTW-GR optimality gap.
4. Call testdata routes official upstream e-VRO/frvcpy reference routes/objectives, not published tours.
5. SOC reserve and Montoya nonlinear results are sensitivity analyses, not main_test sample sizes.
6. With six parents, Holm-adjusted tests have limited power; do not overstate significance.
7. Legacy DDQN is not a paper method.

## Scientific audit (Phase 15)

- 1. Did all five HybridPPO seeds train successfully? Yes. Statuses: {42: 'early_stop', 43: 'completed', 44: 'completed', 45: 'completed', 46: 'early_stop'}.
- 2. What update was selected for each seed? {42: 200, 43: 280, 44: 220, 45: 350, 46: 60}.
- 3. Did any final seed collapse? No selected HybridPPO checkpoint has parent-balanced VAL feasibility of 0.
- 4. Are all final checkpoint losses finite? Yes.
- 5. Is TEST feasibility reported on all 49 routes without filtering? Yes. 49 routes × every paper method; failures retained.
- 6. Does every paper method see the exact same TEST route population? Yes. Audit compared route_id sets.
- 7. Are failures retained in all-routes metrics? Yes. `completion_time_all_routes` is present on every main_test row and uses H for failures.
- 8. Are route-weighted and parent-balanced metrics both available? Yes. Tables A/E and this report.
- 9. Are seed and parent uncertainty both represented? Yes. Hierarchical bootstrap for learned methods; parent-cluster for deterministic; seed mean±SD.
- 10. Are paired tests parent-matched? Yes. Exact sign-flip over six parents; learned side averaged over seeds.
- 11. Is DDQN absent from all paper results? Yes. `ddqn_absent=True`.
- 12. Is frvcpy clearly separated from EVRPTW-GR? Yes. Separate scenario `frvcpy_native`; tables state native FRVCP only.
- 13. Is restricted search clearly marked non-global-exact? Yes. `exact=false`, `exact_for=restricted_continuation_or_full_soc`.
- 14. Is SOC reserve separate from main_test? Yes. `scenario=soc_reserve`; greedy deduplicated.
- 15. Are nonlinear Montoya results clearly marked external sensitivity? Yes. `scenario=nonlinear_sensitivity`; table note states not original EVRPTW-GR.
- 16. Are pilot/smoke numbers absent from paper tables? Yes. Tables read only `results/final/raw/final_*.jsonl`.
- 17. Do corpus and split hashes still match? Yes. {'corpus.jsonl': '1796d817ff058fe0559021ab79ca97087ef3646c57744c2c883ecc37e85451e0', 'manifest.csv': '41bb8be3245f8f3b0800db4ba74cb19a2003236b6fbd0dbc1237e7cdfaee104d', 'corpus_metadata.json': '1ca8725199842f0a8d4a4e6fac2a31931ea3c9aaca833c2d253c6147469d9a96', 'train.json': '3aecd73e9472b8572e899a7e2da3f1f874f0e5a13250f5cd6d76d812002913ce', 'validation.json': '1379aef7e92c308f90ab78674dec9994db72078c96cb1020ecc49d33eef69af4', 'test.json': '9bf39fbf1f27aa57146879fb71922b0216f24f1eb37a4fd9a9a43b44b6c4b6d0', 'split_metadata.json': '4740ab71737bfd9c04d37f258e57bfa4854a526da7c03b218ffe877818128900'}.
- 18. Is there any evidence of TEST-driven tuning? No. Checkpoints frozen before TEST; no retrain/reselect after freeze.
- 19. Are all final numeric claims reproducible from tracked raw records? Yes. Tables/figures/report are generated from `results/final/raw`.
- 20. Are tables and figures generated automatically rather than hand edited? Yes. `make_tables.py` / `make_figures.py` / this script.

## Completeness

The computational study specified in the final experiment protocol is complete: training, checkpoint freeze, TEST evaluation, sensitivity/reference suites, integrity audit, statistics, tables, figures, and tracked archive.

