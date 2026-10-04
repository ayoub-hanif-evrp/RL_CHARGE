# Publication outputs (`results_paper/`)

Single source of truth for the manuscript. Every figure is written as:

- **PDF** — vector version for LaTeX
- **PNG** — high-resolution preview (600 dpi)

## Main paper

| Artifact | Role | Evidence |
|----------|------|----------|
| `figures/fig01_method_schematic.{pdf,png}` | Conceptual FA-HPPO method | schematic (not data) |
| `figures/fig02_main_test.{pdf,png}` | Confirmatory performance | V3 TEST |
| `figures/fig03_effect_sizes.{pdf,png}` | Paired effect sizes | V3 TEST / `paired_primary.json` |
| `figures/fig04_difficulty_robustness.{pdf,png}` | Layout x length robustness | V3 TEST (charging-required) |
| `figures/fig05_mechanism.{pdf,png}` | Gold ablation + amount sensitivity | gold VAL + V3 TEST |
| `tables/table01_*` … `table04_*` | Protocol / main / ablation / amount | see MANIFEST |
| `CAPTIONS.md` | Self-contained figure captions | — |

## Appendix

| Artifact | Role |
|----------|------|
| `figures/appendix/figA01_illustrative_val_trajectory.{pdf,png}` | VAL case study (not TEST) |
| `figures/appendix/figA02_learning_curves.{pdf,png}` | Gold development learning curves |
| `figures/appendix/figA03_training_stability.{pdf,png}` | Value-loss / grad-norm diagnostics |
| `figures/appendix/figA04_envelope_ablation.{pdf,png}` | Post-hoc SynthCharge envelope A/B/C |
| `figures/appendix/figA05_failure_analysis.{pdf,png}` | Frozen-raw failure concentration |
| `figures/appendix/figA06_seed_robustness.{pdf,png}` | Per-seed robustness |
| `figures/appendix/figA07_native_frvcp_reference.{pdf,png}` | Native FRVCP reference |
| `tables/tableA01_*` … `tableA03_*` | Per-seed / paired / failure routes |
| `case_study/illustrative_val_episode.json` | Source for Fig. A1 |
| `failure_analysis/` | Deeper frozen-raw failure write-up |

## Regenerate

```bash
python scripts/paper/record_illustrative_val_episode.py   # once (VAL case study JSON)
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Does **not** retrain or re-evaluate the consumed V3 TEST.

`scripts/paper/build_paper_artifacts.py` is deprecated and delegates here.
