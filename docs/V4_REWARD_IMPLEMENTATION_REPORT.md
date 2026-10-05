# V4 reward implementation report

**Date:** 2026-10-05  
**Stage:** V4-F reward finalization — selected `V4_BASE_NO_L_FAIL`  
**Confirmatory TEST consumed:** **No**

---

## 1. Selected reward (normalized time-horizon)

Feasible transitions:

\[
r_t=-\frac{\Delta t}{C_{\mathrm{train}}}
\]

Terminal failure:

\[
r_{\mathrm{fail}}=-\frac{H-t}{C_{\mathrm{train}}}
\]

Episode return:

\[
G=
\begin{cases}
-T_{\mathrm{completion}}/C_{\mathrm{train}}, & \text{successful route}\\
-H/C_{\mathrm{train}}, & \text{failed route}
\end{cases}
\]

with \(C_{\mathrm{train}}=10\) on SynthCharge.

## 2. PBRS ablation (not selected)

`V4_PBRS` remains an **ablation**, not the selected reward.

- Mean parent-balanced VAL feasibility did **not** improve (97.8% vs 98.2%).
- Correct unshaped comparator: **`V4_BASE_NO_L_FAIL`** (same base failure without \(L\), plus shaping).
- Negative/neutral PBRS result is retained in `SUMMARY` / figures and is not overwritten.
- No new reward weights were introduced after seeing VAL results.

## 3. Exact \(C_{\mathrm{train}}\)

| Field | Value |
|-------|-------|
| Rule | `median_depot_horizon_over_TRAIN_routes_only` |
| Value | **10.0** |
| File | `results/v4_reward/C_TRAIN.json` |

On SynthCharge TRAIN, median = max = 10, so `V4_BASE` is numerically equivalent to `V3_TIME` with PPO `return_scale=10`.

## 4. Logging / gamma safety

PPO metrics (see `src/rl/ppo.py`):

| Metric | Meaning |
|--------|---------|
| `objective_episode_return_mean` | unnormalized \(-T\) / \(-H\) |
| `normalized_base_episode_return_mean` | objective / \(C_{\mathrm{train}}\) |
| `shaping_episode_return_mean` | accumulated PBRS shaping |
| `training_episode_return_mean` | actual PPO reward return |
| `raw_episode_return_mean` | **legacy alias** of training return (marked for compatibility) |

For PBRS, `RewardConfig.assert_compatible_with_ppo_gamma` requires `ppo_config.gamma == reward_config.gamma` (tol \(10^{-12}\)).

## 5. Exact-horizon boundary

If a successful route finishes at \(T=H\), success and failure returns coincide for `V4_BASE_NO_L_FAIL`.  
Audit artifacts: `results/v4_reward/analysis/EXACT_HORIZON_AUDIT.json` (and paired-analysis horizon section).  
No epsilon penalty was added; empirical TRAIN/VAL successes under development checkpoints are documented there.

## 6. Development ablation (4 × 5, dirty-tree runs — not final freeze)

| Variant | VAL feas. mean±SD | Completion mean±SD | Mean best update |
|---|---:|---:|---:|
| V3_TIME | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |
| V4_BASE | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |
| V4_PBRS | 97.8% ± 1.1 | 3.891 ± 0.082 | 222 |
| V4_BASE_NO_L_FAIL | 98.2% ± 1.0 | 3.878 ± 0.046 | 292 |

Source: `results/v4_reward/SUMMARY.md`. These checkpoints are **development evidence only**; final freeze uses `final_clean/`.

### Lexicographic selection

1. Maximize mean parent-balanced VAL feasibility → tie `V3_TIME` / `V4_BASE` / `V4_BASE_NO_L_FAIL` at 98.2%.
2. Among tied, minimize mean parent-balanced failure-retaining completion → **`V4_BASE_NO_L_FAIL`** (3.878).
3. Secondary energy/station/SOC metrics reported on common-feasible matched pairs only; they do not redefine the objective.

## 7. Per-seed feasibility (development)

- **V3_TIME**: 42:98.9%, 43:97.8%, 44:95.6%, 45:98.9%, 46:100.0%
- **V4_BASE**: 42:98.9%, 43:97.8%, 44:95.6%, 45:98.9%, 46:100.0%
- **V4_PBRS**: 42:98.9%, 43:96.7%, 44:98.9%, 45:97.8%, 46:96.7%
- **V4_BASE_NO_L_FAIL**: 42:98.9%, 43:98.9%, 44:98.9%, 45:97.8%, 46:96.7%

## 8. Route-level VAL + paired analysis

- Export: `results/v4_reward/val_route_rows/` (90 routes × 4 variants × 5 seeds; no TEST).
- Analysis: `results/v4_reward/analysis/` (paired win/tie/loss, common-feasible operational deltas).

## 9. Figures

Rebuild (selected reward = `V4_BASE_NO_L_FAIL`; prefers `final_clean` curves when present):

```bash
python scripts/paper/build_v4_training_figures.py
python scripts/v4_reward/build_comparison_figure.py
```

- Aggregation: common update support across seeds; no forward-fill beyond last observed update.
- Uncertainty: Student-t 95% CI (n=5 ⇒ \(t_{0.975,4}\)).
- Ablation figure: full 0–105% feasibility axis; seed points + mean.

## 10. Final clean freeze

After code/docs fixes are committed on a clean tree:

```bash
python scripts/v4_reward/run_final_clean.py
```

Artifacts:

- `results/v4_reward/final_clean/V4_BASE_NO_L_FAIL/seed_{42..46}/`
- `checkpoints_v4/final_reward/V4_BASE_NO_L_FAIL/seed_{42..46}/`
- `results/v4_reward/FINAL_REWARD_FREEZE.json`

Manifests require `git_dirty=false`, repository-relative checkpoint paths, TRAIN/VAL hashes, PPO config hash, checkpoint SHA256.

## 11. V1–V3 integrity

V1/V2/V3 locks, raw results, and checkpoints were not modified for this finalization.  
**No V4 confirmatory TEST was generated or consumed.**

## 12. Methodology statement

FA-HPPO uses an objective-aligned time reward rather than a weighted mixture of
arbitrary penalties. Feasible transitions incur normalized elapsed-time cost,
while failure maps the episode to the route horizon. Hard EV and time-window
requirements are enforced structurally. Distance contributes through travel
time, while energy charged, charging stops, and terminal SOC are reported
separately as operational metrics.
