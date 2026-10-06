# SynthCharge B0/B1/B3/B2 same-domain ablation

**POST-HOC same-domain development robustness analysis — NOT V3 CONFIRMATORY.**

Reproduces the gold time-cap × return-scaling 2×2 on SynthCharge TRAIN/VAL.

| Variant | Time-aware cap | Return scale |
|---------|----------------|--------------|
| B0 | off | off |
| B1 | on | off |
| B3 | off | on |
| B2 | on | on |

Seeds: 42–46. Budget: `configs/common/rl/hybrid_ppo.toml`.

**Status:** script prepared; full 20-cell execution not started in the scientific-fixes pack (~15–22 CPU-hours).

```bash
python scripts/development/run_synthcharge_b_ablation.py --variant all
```
