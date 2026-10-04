# Publication figure redesign report

Latest polish after editorial review of `3ccb8c1`.

## Hierarchy (current)

### Main (6 figures — one scientific question each)

| Fig | Stem | Question |
|-----|------|----------|
| 1 | `fig01_method_schematic` | What is FA-HPPO? |
| 2 | `fig02_main_test` | Does it work on locked V3 TEST? |
| 3 | `fig03_effect_sizes` | How large are paired advantages? |
| 4 | `fig04_difficulty_robustness` | Across which regimes? |
| 5 | `fig05_development_ablation` | Which gold components matter? (development) |
| 6 | `fig06_amount_sensitivity` | Does learned u beat Max on TEST? |

### Appendix (9 figures)

| Fig | Stem |
|-----|------|
| A1 | `figA01_illustrative_trajectory` (route + SOC) |
| A2 | `figA02_illustrative_hybrid_decision` |
| A3 | `figA03_learning_curves` |
| A4 | `figA04_training_stability` |
| A5 | `figA05_envelope_ablation` (paired seeds; zoomed axis labeled) |
| A6 | `figA06_failure_consistency` (R01–R20) |
| A7 | `figA07_failure_heatmap` |
| A8 | `figA08_seed_robustness` |
| A9 | `figA09_native_frvcp_reference` |

## Editorial fixes in this pass

1. Split former Fig. 5 into independent Fig. 5 (gold ablation) and Fig. 6 (TEST amount).
2. Split former A1 into trajectory (A1) and hybrid decision (A2).
3. Split failure lollipop (A6) and heatmap (A7); short aliases R01–R20 + Table A3.
4. Fig. 1: leave-and-rejoin charge detour; staggered SOC labels; explicit \(u=0.82\).
5. Fig. 2: removed redundant legend; numeric completion labels.
6. Fig. 3: printed effect estimates beside points.
7. Fig. 4: sequential 0→max scale for nonnegative Δ (not awkward tiny-negative diverging).
8. A5: paired seed trajectories; “Zoomed y-axis” labeled; no lone `(a)`.
9. A8: fixed overlapping truncated-axis / CI annotations.
10. A3/A4: removed lone `(a)` on single-panel learning curves; shared legend on training stability.

## Integrity

- No V3 TEST replay / retrain / src change.
- `build_results_paper.py --verify` → 15 PNG + 15 PDF.
- Full pytest green.

## Pipeline

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```
