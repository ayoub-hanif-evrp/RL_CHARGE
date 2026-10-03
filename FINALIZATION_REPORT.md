# Finalization report — FA-HPPO / V3-HPPO paper artifact

## 1. Starting HEAD

`557f1b38228d6d44fca34331805ecfde4f3d2fed`

## 2. Ending HEAD

See latest `main` after the ablation-complete commit (starts from `e527e06` TEST archive).

## 3. Tests run and status

```text
python -m pytest tests -q
230 passed
```

(Previously 226; +4 from `tests/test_v3_hppo_protocol.py`.)

## 4. Files removed

None (soft cleanup only).

## 5. Files moved

None of frozen V1/V2 scientific paths. Paper TEST generator config lives under
`results/v3_hppo/configs/` (not under frozen `configs/`) so the method freeze
tree stays intact.

## 6. Files archived / preserved

| Path | Role |
|------|------|
| `results/v2/final/` | Untouched historical HybridPPO+DiscretePPO freeze |
| `results/final/` | Untouched V1 archive |
| `results/v2/diagnostics/` | Historical two-seed B0–B3 diagnostics |

## 7. Files added (high level)

- `README.md`, `REPRODUCIBILITY.md`, `RESULTS_INDEX.md`, `RELEASE_CHECKLIST.md`
- `docs/CLEANUP_MANIFEST.md`, `docs/ARCHIVE_LAYOUT.md`
- `paper/` (CLAIMS, RESULTS_MAP, figures, tables)
- `results/v3_hppo/` protocol, freezes, locks, raw, statistics, ablation
- `data/routes_v2/synthcharge_v3_test/`, `data/splits_v2/synthcharge_v3_test.json`
- `scripts/v3_hppo/`, `scripts/paper/build_paper_artifacts.py`
- `tests/test_v3_hppo_protocol.py`

## 8. New V3-HPPO protocol

`results/v3_hppo/PAPER_PROTOCOL.json`

- Method: FA-HPPO / HybridPPO B2 only (no DiscretePPO)
- Seed start: **400000** (audit: prior SynthCharge seeds ≤ 300550)
- Hypotheses H1–H4 predeclared
- Amount ablations FA-HPPO-Min/Max predeclared

## 9. New TEST seed range and hashes

- Seed range used: **400000–402845** (accepted seeds only; candidates audited)
- Routes: **180** (9 cells × 20; 16 charging-required + 4 no-charge each)
- Lock: `results/v3_hppo/TEST_LOCK.json` (187 files)
- Split: `data/splits_v2/synthcharge_v3_test.json`

## 10. Checkpoint hashes

`results/v3_hppo/CHECKPOINT_FREEZE.json` — 5 SynthCharge HybridPPO `best.pt` hashes,
cross-checked identical to V2 freeze (no reselection).

## 11. TEST row count

**3240** rows = 180 × (5 HybridPPO + 5 Min + 5 Max + 3 baselines)

`EVALUATION_CONSUMED.json` written; evaluation git SHA `5c791f5…`.

## 12. Integrity-audit result

`python scripts/paper/build_paper_artifacts.py --verify` → **ok**  
No DiscretePPO in V3 matrix; physics profile `synthcharge_linear` only.

## 13. Final FA-HPPO results (V3 TEST)

| Metric | Value |
|--------|-------|
| Feasibility (mean across 5 seeds) | **0.942** |
| 95% hierarchical bootstrap CI | **[0.920, 0.960]** |
| Failure-retaining completion | **4.2** |
| Dominant failure reason | `NO_FEASIBLE_ACTION` (52 route×seed) |

## 14. Baseline results (same TEST)

| Method | Feasibility | Completion |
|--------|-------------|------------|
| OneStepLookahead | 0.700 | 5.3 |
| GreedyFull | 0.467 | 6.9 |
| GreedyMin | 0.394 | 7.3 |

## 15. Ablation results

### Amount ablations (eval on frozen FA-HPPO; V3 TEST)

| Method | Feasibility |
|--------|-------------|
| FA-HPPO (free \(u\)) | 0.942 |
| FA-HPPO-Max (\(u=1\)) | 0.943 |
| FA-HPPO-Min (\(u=0\)) | 0.079 |

Min collapsing is reported honestly; Max ≈ free amount on this TEST.

### B0/B1/B2/B3 five-seed gold TRAIN/VAL

**Complete** (20/20): seeds 42–46 under `results/v3_hppo/ablation/`.

| Variant | Time cap | Return scale | Mean parent-bal. VAL feas | Mean value loss |
|---------|----------|--------------|---------------------------|-----------------|
| B0 | off | off | 0.467 | ~3.3e5 |
| B1 | on | off | 0.554 | ~3.1e5 |
| B3 | off | on | 0.801 | ~0.0019 |
| B2 (full FA-HPPO) | on | on | **0.900** | ~0.0017 |

Development/validation only — not fresh TEST evidence.
Historical two-seed diagnostics remain in `results/v2/diagnostics/`.

## 16. Generated paper figures

Under `paper/figures/`:

- fig01_method_overview (pdf/svg/png)
- fig02_main_test
- fig03_charging_required
- fig04_seed_robustness
- fig05_ablation
- fig06_training_stability
- fig07_difficulty_heatmap
- fig08_failure_analysis
- fig09_native_frvcp_reference

## 17. Generated paper tables

Under `paper/tables/`: Markdown + CSV + LaTeX for Tables 1–5 and Appendix per-seed.

## 18. Known limitations

- SynthCharge linear energy ≠ EVRPTW-GR terrain/payload physics.
- FA-HPPO does not choose customer order.
- Certificate timeout ≠ infeasibility proof.
- frvcpy is a native FRVCP reference, not an EVRPTW-GR oracle.
- Fig 9 points at archived V1 FRVCP figure rather than a newly recomputed plot.

## 19. Owner TODOs

See `RELEASE_CHECKLIST.md`: license, CITATION.cff authors/ORCID, Zenodo DOI after tag.

## 20. Regenerate paper artifacts (no TEST rerun)

```bash
python scripts/paper/build_paper_artifacts.py
python scripts/paper/build_paper_artifacts.py --verify
```

Do **not** re-run `evaluate_paper_test.py --execute-once` (consumed).
