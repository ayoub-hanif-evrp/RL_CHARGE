# Publication outputs (`results_paper/`)

Single source of truth for manuscript figures (PNG only) and tables.

## Main paper

| Artifact | Role | Source | Split | Seeds | CI |
|----------|------|--------|-------|-------|-----|
| `figures/fig01_main_test.png` | Confirmatory performance | `results/v3_hppo/raw/synthcharge_test.jsonl` | V3 TEST | 5 | hierarchical bootstrap |
| `figures/fig02_effect_sizes.png` | Paired effect sizes | `statistics/paired_primary.json` | V3 TEST | 5 | predeclared paired CI |
| `figures/fig03_ablation_and_training_stability.png` | Methodology | `results/v3_hppo/ablation/` | gold VAL | 5 | seed scatter |
| `figures/fig04_amount_sensitivity.png` | Amount policy | V3 raw | V3 TEST | 5 | hierarchical bootstrap |
| `figures/fig05_difficulty_and_failures.png` | Difficulty / failures | V3 raw | V3 TEST | 5 | cell means |
| `tables/table01_*` | Protocol | metadata | — | — | — |
| `tables/table02_*` | Main results | V3 raw/stats | V3 TEST | 5 | hierarchical bootstrap |
| `tables/table03_*` | Ablation | ablation JSON | gold VAL | 5 | SD across seeds |
| `tables/table04_*` | Amount sensitivity | V3 raw | V3 TEST | 5 | hierarchical bootstrap |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_seed_robustness.png` | Per-seed robustness |
| `figures/appendix/figA02_native_frvcp_reference.png` | Native FRVCP (not EVRPTW-GR / not SynthCharge) |
| `tables/tableA01_*` | Per-seed FA-HPPO |
| `tables/tableA02_*` | Primary paired statistics |

## Regenerate

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate TEST.

## Notes

- FA-HPPO CIs: hierarchical bootstrap resampling training seed then route.
- Deterministic baselines: no fabricated seed uncertainty.
- Fig 3 is **development / gold validation**, not confirmatory TEST.
- V3 = fresh independently generated SynthCharge TEST (not external-domain generalization).
