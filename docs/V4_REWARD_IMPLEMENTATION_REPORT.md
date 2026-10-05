# V4 reward implementation report

**Date:** 2026-10-05  
**Stage:** V4-F reward study — full TRAIN/VAL ablation complete  
**Confirmatory TEST consumed:** **No**

---

## 1. Exact mathematical reward equation

Feasible step (`V4_PBRS`):

\[
r_t
=
-\frac{\Delta t}{C_{\mathrm{train}}}
+
\Phi(s_{t+1})-\Phi(s_t),
\qquad
\Phi(s)=-\frac{L_{\mathrm{remaining}}(s)}{C_{\mathrm{train}}},
\quad
\Phi_{\mathrm{absorbing}}=0.
\]

## 2. Exact failure equation (`V4_PBRS`)

\[
r_{\mathrm{fail}}^{\mathrm{base}}=-\frac{H-t_0}{C_{\mathrm{train}}},
\qquad
F=\frac{L(s_t)}{C_{\mathrm{train}}}.
\]

Cumulative shaped failure return: \((L_0-H)/C_{\mathrm{train}}\).

## 3. Exact \(C_{\mathrm{train}}\)

| Field | Value |
|-------|-------|
| Rule | `median_depot_horizon_over_TRAIN_routes_only` |
| Value | **10.0** |
| File | `results/v4_reward/C_TRAIN.json` |

On SynthCharge TRAIN, median = max = 10, so `V4_BASE` is numerically equivalent to `V3_TIME` with PPO `return_scale=10`.

## 4. Files added / changed

See previous section of this report / git history. Main runtime artifacts now populated under:

- `results/v4_reward/ablation/{variant}/seed_{42..46}/`
- `checkpoints_v4/reward_ablation/{variant}/seed_{42..46}/`
- `results_v4/figures/`

## 5. Tests

Full suite previously green (252). Reward-specific: `tests/rl/test_v4_rewards.py`.

## 6. V1–V3 integrity

V3 locks/raw/checkpoints untouched. No V4 TEST consumed.

## 7–9. TRAIN/VAL experiments (full 4 × 5)

All **20 cells** completed (~5.2 h wall-clock, 5-wide parallel).

| Variant | VAL feas. mean±SD | Completion mean±SD | Mean best update |
|---|---:|---:|---:|
| V3_TIME | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |
| V4_BASE | 98.2% ± 1.7 | 3.915 ± 0.123 | 220 |
| V4_PBRS | 97.8% ± 1.1 | 3.891 ± 0.082 | 222 |
| V4_BASE_NO_L_FAIL | 98.2% ± 1.0 | 3.878 ± 0.046 | 292 |

Source: `results/v4_reward/SUMMARY.md`.

### Interpretation (development only)

1. **`V3_TIME` ≡ `V4_BASE`** on this corpus (identical per-seed numbers): expected, because \(C_{\mathrm{train}}=10\) matches V3 PPO scale.
2. **`V4_PBRS`**: slightly lower mean feasibility (−0.4 pp) but slightly better mean completion and lower seed SD. Not a clear win on the primary lexicographic VAL criterion.
3. **`V4_BASE_NO_L_FAIL`**: same mean feasibility as V3/BASE, **best mean completion** and tightest completion SD; took longer on average to select best ckpt (292 updates). Suggests the \(L_{\mathrm{remaining}}\) failure-progress term is **not clearly helpful** on SynthCharge VAL under this protocol.

Negative/neutral finding is reported as required: PBRS did not dominate V3/BASE on parent-balanced VAL feasibility.

## 8. Per-seed feasibility

- **V3_TIME**: 42:98.9%, 43:97.8%, 44:95.6%, 45:98.9%, 46:100.0%
- **V4_BASE**: 42:98.9%, 43:97.8%, 44:95.6%, 45:98.9%, 46:100.0%
- **V4_PBRS**: 42:98.9%, 43:96.7%, 44:98.9%, 45:97.8%, 46:96.7%
- **V4_BASE_NO_L_FAIL**: 42:98.9%, 43:98.9%, 44:98.9%, 45:97.8%, 46:96.7%

## 10–12. Figures (from saved curves)

- `results_v4/figures/fig_v4_reward_evolution.png` (V4_PBRS, 5 seeds)
- `results_v4/figures/fig_v4_loss_evolution.png`
- `results_v4/figures/fig_v4_validation_learning.png`
- `results_v4/figures/fig_v4_gradient_norm.png`
- `results_v4/figures/fig_v4_reward_ablation_val.png` (4-variant VAL bars)

Rebuild:

```bash
python scripts/paper/build_v4_training_figures.py
python scripts/v4_reward/build_comparison_figure.py
```

## 13. Unresolved scientific issues

1. Reward selection is **not frozen** yet — VAL evidence does not uniquely favor PBRS.
2. Common-feasible energy/visit summaries across variants not yet exported as tables.
3. Fresh V4 TEST **not authorized** until an explicit freeze + TEST protocol.

## 14. Explicit TEST statement

**No new confirmatory TEST was generated or consumed.**  
Selection remains TRAIN/VAL development evidence only.
