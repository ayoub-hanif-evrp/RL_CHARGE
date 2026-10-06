# Repository cleanup + publication-figure redesign

HEAD before this pass: `cae4431`

## Classification summary

| Class | Examples | Action |
|-------|----------|--------|
| CURRENT_CORE | `src/`, `scripts/v3_hppo/`, `configs/`, tests, frozen `results/v3_hppo/` | retained |
| CURRENT_PAPER | `results_paper/`, `scripts/paper/build_results_paper.py`, `paper/` | redesigned |
| HISTORICAL_PROVENANCE | `results/final/`, `results/v2/`, data trees, V1/V2 scripts | retained in place |
| DEAD/REDUNDANT | empty `results/figures|tables` gitkeeps; deprecated `build_paper_artifacts.py`; overlapping status reports | removed/moved |

## Deleted

- `scripts/paper/build_paper_artifacts.py` — deprecated forwarder; no unique logic
- `results/figures/.gitkeep`, `results/tables/.gitkeep` — empty namespaces superseded by `results_paper/`

## Moved to `docs/history/`

- `FINALIZATION_REPORT.md`
- `docs/FIGURE_REDESIGN_REPORT.md`
- `docs/FINAL_CLEANUP_REPORT.md`
- `docs/FINAL_CLEANUP_AUDIT.md`
- `docs/SCIENTIFIC_REPAIR_REPORT.md`
- `docs/CLEANUP_MANIFEST.md`

Active scientific notes (`STATION_FEATURE_AUDIT`, `CONTINUOUS_HEAD_LIMITATION`, `V4_PROPOSAL`, `SCIENTIFIC_FIXES_REPORT`, etc.) remain under `docs/`.

## Figure redesign (PNG only)

Main:

1. `fig01_usecase_route` — real VAL route map
2. `fig02_main_feasibility` — horizontal bars
3. `fig03_difficulty_robustness` — heatmaps
4. `fig04_development_interaction` — 2×2 interaction
5. `fig05_amount_route_comparison` — learned/tie/Max stacked bar

Appendix A1–A10: SOC, paired effects, completion, curves, stability, envelope, failures, seeds, FRVCP.

No prose annotations inside plots; captions in `results_paper/CAPTIONS.md`.
No PDF figure outputs.

## Preserved unchanged

- V3 raw / `TEST_LOCK` / `EVALUATION_CONSUMED` / checkpoints
- V1 `results/final/`, V2 `results/v2/`
- Frozen data under `data/`

## Validation

- `build_results_paper.py --verify` → ok (`n_png=15`, `n_pdf=0`)
- V3 hashes unchanged
- Full pytest expected green after this commit
