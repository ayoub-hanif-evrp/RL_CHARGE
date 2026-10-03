# Final cleanup report

## Starting HEAD

`5c6a5968707155bf13152ef889be54510bf5d7fb`

## Ending HEAD

`1a2b606` (tip of `main`; publication milestone `9b55be4`)

## Files / folders deleted (summary)

| Item | Count / note |
|------|----------------|
| Unlocked V3 candidate instances | **2666** removed; **180** locked TEST instances kept |
| `results/smoke/` | entire smoke table dump |
| `results/pilot/**` except `correctness_audit/official_route_plan_mapping.json` | development pilots |
| `paper/figures/`, `paper/tables/` | duplicate generated paper outputs (replaced by `results_paper/`) |

## Files moved

None (publication outputs newly created under `results_paper/`).

## Approximate reduction

| Metric | Before (`5c6a596`) | After cleanup |
|--------|--------------------|---------------|
| Tracked files | ~7320 | ~4507 |
| Δ tracked files | | **≈ −2813** |
| V3 TEST instances | 2846 | **180** |

Repository object size reduction is dominated by the deleted candidate `.txt` instances (recoverable from git history).

## Rejected V3 candidates removed

**2666** unlocked instance files under `data/routes_v2/synthcharge_v3_test/instances/`.  
`candidate_audit.jsonl` retained. TEST_LOCK verified for 180 members.

## Preserved frozen paths

- `results/v3_hppo/` protocol, locks, raw, statistics, ablation, configs, wording
- `results/v2/final/` (includes DiscretePPO)
- `results/final/` historical V1 / FRVCP archive
- Scientific freeze trees (`src`, `third_party`, core configs)

## Final `results_paper/` tree

```
results_paper/
  README.md
  MANIFEST.json
  figures/
    fig01_main_test.png
    fig02_effect_sizes.png
    fig03_ablation_and_training_stability.png
    fig04_amount_sensitivity.png
    fig05_difficulty_and_failures.png
    appendix/
      figA01_seed_robustness.png
      figA02_native_frvcp_reference.png
  tables/
    table01_benchmark_protocol.{csv,md,tex}
    table02_main_results.{csv,md,tex}
    table03_development_ablation.{csv,md,tex}
    table04_amount_sensitivity.{csv,md,tex}
    tableA01_per_seed.{csv,md,tex}
    tableA02_primary_statistics.{csv,md,tex}
```

Figures: **PNG only** (no PDF/SVG).

## Tests / verification

- `python -m pytest tests -q` → **232 passed**
- `python scripts/paper/build_results_paper.py --verify` → **ok**
- Method freeze diff empty
- Raw TEST hash matches `EVALUATION_CONSUMED`
- TEST_LOCK verified; 180/180 instances
- No TEST rerun

## Owner-only remaining

License, CITATION.cff, Zenodo/DOI, manuscript `.tex` (see `RELEASE_CHECKLIST.md`).
