# Publication outputs (`results_paper/`)

PNG figures only. Single source of truth for the manuscript.

## Main paper

| Artifact | Role | Evidence |
|----------|------|----------|
| `figures/fig01_method_case_study.png` | Method illustration (route + SOC-along-route + hybrid decision) | **VAL** case study JSON — not TEST |
| `figures/fig02_main_test.png` | Confirmatory performance | V3 TEST |
| `figures/fig03_effect_sizes.png` | Paired effect sizes | V3 TEST / `paired_primary.json` |
| `figures/fig04_ablation_and_training_stability.png` | Methodology components | gold VAL ablation |
| `figures/fig05_amount_sensitivity.png` | Amount-policy sensitivity | V3 TEST |
| `tables/table01_*` … `table04_*` | Protocol / main / ablation / amount | see MANIFEST |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_difficulty_heatmap.png` | Cell feasibility + station visits |
| `figures/appendix/figA02_seed_robustness.png` | Per-seed robustness |
| `figures/appendix/figA03_native_frvcp_reference.png` | Native FRVCP reference |
| `figures/appendix/figA04_learning_curves.png` | Development VAL learning curves |
| `tables/tableA01_*`, `tableA02_*` | Per-seed / paired stats |
| `case_study/illustrative_val_episode.json` | Source for Fig. 1 |

## Regenerate

```bash
python scripts/paper/record_illustrative_val_episode.py
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate the consumed V3 TEST.
