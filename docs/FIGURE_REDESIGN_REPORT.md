# Publication figure redesign report

Date: 2026-10-04  
HEAD at start: `310ff6fa3ea4410cfb3a188d31f19404d7a752f0`

## 1. Files changed

- `scripts/paper/build_results_paper.py` — sole publication builder (PDF+PNG; new figure hierarchy)
- `scripts/paper/build_paper_artifacts.py` — deprecated; delegates to official builder
- `scripts/paper/build_failure_analysis.py` — analysis JSON/README only; figure via official builder
- `tests/test_v3_hppo_protocol.py` — allow PDF+PNG; new figure names
- `paper/RESULTS_MAP.md`, `results_paper/README.md`, `results_paper/CAPTIONS.md`, `results_paper/MANIFEST.json`
- `REPRODUCIBILITY.md`, `RESULTS_INDEX.md`
- Regenerated `results_paper/figures/**` and tables

## 2. Figures replaced (main)

| New | Role |
|-----|------|
| `fig01_method_schematic` | Conceptual FA-HPPO method (3 panels) |
| `fig02_main_test` | Horizontal confirmatory TEST feasibility + completion |
| `fig03_effect_sizes` | Polished paired forest plot |
| `fig04_difficulty_robustness` | Layout×length feasibility + Δ vs Lookahead |
| `fig05_mechanism` | Gold B0/B1/B3/B2 + TEST amount sensitivity |

## 3. Figures moved / renumbered (appendix)

| New | Former role |
|-----|-------------|
| `figA01_illustrative_val_trajectory` | Former main Fig.1 VAL case study |
| `figA02_learning_curves` | Former A04 |
| `figA03_training_stability` | Split from former main Fig.4 diagnostics |
| `figA04_envelope_ablation` | New post-hoc SynthCharge A/B/C panel |
| `figA05_failure_analysis` | Redesigned former A05 |
| `figA06_seed_robustness` | Former A02 |
| `figA07_native_frvcp_reference` | Former A03 |

Obsolete stems removed: `fig01_method_case_study`, `fig04_ablation_and_training_stability`, `fig05_amount_sensitivity`, old appendix names.

## 4. Scientific values used (from frozen sources)

- V3 TEST main: FA-HPPO 0.942 / Lookahead 0.700 / Greedy Full 0.467 / Greedy Min 0.394
- Amount sensitivity: FA-HPPO 0.942 / Max 0.943 / Min 0.079; 173/180 ties (from `amount_sensitivity.json`)
- Gold ablation means loaded from `results/v3_hppo/ablation/*/validation.json`
- Envelope A/B/C from `results/development/envelope_ablation/SUMMARY.json` (~99.3 / 98.9 / 98.2%)
- Failure analysis: 52 route×seed fails; 20 problematic routes; 160/180 all-seed OK; all reasons `NO_FEASIBLE_ACTION`

## 5. No V3 TEST replay / retraining

Confirmed: builder only reads frozen raw/statistics/ablation/envelope archives and VAL case-study JSON. No policy evaluation, no checkpoint training, no TEST membership changes.

## 6. Frozen integrity

- `raw_sha256_lf` matches `EVALUATION_CONSUMED.json`
- `TEST_LOCK.json` clean (0 mismatches)
- `build_results_paper.py --verify` → ok (`n_png=12`, `n_pdf=12`)

## 7. Could not produce from frozen data

- Quality-vs-runtime vs a strong same-problem planner: no such comparator exists yet (documented previously; not invented).
- Step-level pre-failure TEST trajectories: not stored; failure figure uses frozen route×seed outcomes only.

## Pipeline

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```
