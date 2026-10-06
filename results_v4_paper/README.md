# V4 paper-facing package (standalone)

This directory contains **only** V4 manuscript-facing evidence.
It does not compare to prior experimental versions.

## Primary TEST result

FA-HPPO achieves **94.56%** [92.30, 96.81] feasibility
(95% Student-t CI across five seeds) on the fresh independently generated
held-out SynthCharge TEST and substantially outperforms One-step lookahead,
Greedy full charge, and Greedy minimum charge under the same feasibility-aware
environment.

## Recommended main-paper figures

Use a compact subset in the manuscript; keep the rest as a figure library / supplement.

| Priority | Figure | Role |
|---|---|---|
| Main | Fig. 1 `fig01_usecase_route.png` | Fixed-route charging problem |
| Main | Fig. 2 `fig02_method_schematic.png` | FA-HPPO method schematic |
| Main | Fig. 3 `fig03_soc_envelope.png` | SOC envelope |
| Main | Fig. 5 `fig05_main_feasibility.png` | Main TEST feasibility |
| Main | Fig. 6 or Fig. 8 | Completion / distribution |
| Main | Fig. 11 `fig11_difficulty_heatmap.png` | Difficulty heatmap |
| Main | Fig. 12 `fig12_reward_ablation_val.png` | Reward ablation |
| Optional main | Fig. 4 `fig04_illustrative_soc.png` | Real VAL SOC trajectory (if space) |
| Supplement | Figs. 7, 9, 10, 13–15, A1, A3, A4 | Charging subset, strata, training, diagnostics |

Do not force all 18 figures into the main manuscript.

## Reproduce displays (no TEST rerun / no retraining)

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

Illustrative VAL trajectory source (Fig. 4):
`results_v4_paper/data/illustrative_val_trajectory.json`
(regenerate with `python scripts/paper/record_v4_illustrative_val_trajectory.py` only if needed; do not retrain).

## Benchmark strata

Balanced quotas are **layout × frozen-route-length bin** (20 routes/cell).
See `docs/V4_BENCHMARK_WORDING.md`.

## Claims / limitations

See `paper/V4_CLAIMS.md`.
