# V4 reward study protocol (TRAIN/VAL only)

See `docs/V4_REWARD_DESIGN.md` and `PAPER_PROTOCOL.json`.

## Launch

```bash
python scripts/v4_reward/run_reward_ablation.py --variant all
python scripts/paper/build_v4_training_figures.py
```

## Hard rules

1. Do not touch `results/v3_hppo/`, `results/v2/`, `results/final/`.
2. Do not use consumed V3 TEST for selection or tuning.
3. Do not mix reward changes with V4-A/B/C initially.
4. Report negative/neutral findings.
5. No confirmatory TEST until freeze + dedicated protocol.
