# V4 reward study protocol (TRAIN/VAL only)

See `docs/V4_REWARD_DESIGN.md` and `PAPER_PROTOCOL.json`.

**Selected development reward:** `V4_BASE_NO_L_FAIL` (normalized time-horizon).  
`V4_PBRS` remains a reported ablation only.

## Launch

```bash
# Development ablation (already completed; do not overwrite)
python scripts/v4_reward/run_reward_ablation.py --variant all

# Route-level VAL reeval + paired analysis (no TEST)
python scripts/v4_reward/export_val_route_rows.py
python scripts/v4_reward/audit_exact_horizon.py
python scripts/v4_reward/analyze_val_pairs.py

# Figures (prefer final_clean curves when present)
python scripts/paper/build_v4_training_figures.py V4_BASE_NO_L_FAIL
python scripts/v4_reward/build_comparison_figure.py

# Clean final freeze (requires clean git tree)
python scripts/v4_reward/run_final_clean.py
python scripts/v4_reward/write_final_freeze.py
```

## Hard rules

1. Do not touch `results/v3_hppo/`, `results/v2/`, `results/final/`.
2. Do not use consumed V3 TEST for selection or tuning.
3. Do not mix reward changes with V4-A/B/C initially.
4. Report negative/neutral findings (including PBRS).
5. No confirmatory TEST until freeze + dedicated protocol.
6. Dirty-tree ablation checkpoints are not final freeze checkpoints.
