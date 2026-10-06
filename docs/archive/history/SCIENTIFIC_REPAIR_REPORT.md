# Scientific repair report

**Date context:** post-`aadfa18` repair pass  
**Method freeze:** `df35705c012b4ce88674a5b8906176c97838b06c`  
**Consumed V3 TEST:** unchanged (no retrain, no re-eval, no statistic rewrite)

---

## Summary table

| Issue | Verified evidence | Action taken | Scientific behavior changed? | Retrain? | TEST touched? |
|-------|-------------------|--------------|------------------------------|----------|---------------|
| Station feature index 6 duplicates index 9 | `features.py::_station_features` sets `arrival = interval.soc_lower` | Documented in `docs/FROZEN_METHOD_NOTES.md`; provenance + semantic tests | No | No | No |
| Beta uses softplus+1 (α,β>1) | `policy.py` | Documented carefully; V4 candidates listed | No | No | No |
| Amount-control overselling risk | FA-HPPO ≈ Max; 173/180 tied | Strengthened `paper/CLAIMS.md` + table notes | No (wording) | No | No |
| Cryptic B0/B1/B3/B2 names | Paper tables/figures | Descriptive names primary, B labels kept | Display only | No | No |
| Missing lower-bound ablation | B\* varies time×scale only | Script+protocol under `scripts/development/` + `results/development/envelope_ablation/` | No runs claimed as confirmatory | Not in this pass | No |
| Weak baselines vs planning | Greedy / 1-step | `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md`; prefer V4 TEST | No | No | No |
| Failure analysis incomplete | 52 fail rows in frozen raw | `tableA03_*`, `figA05_*` from frozen raw only | Display only | No | No |
| p-value false precision | MC floor ≈5e-5 | Paper-facing `<5e-5`; raw JSON unchanged | Display only | No | No |
| Environment lock incomplete | Partial env JSON only | `docs/ENVIRONMENT_REPRODUCIBILITY.md`, partial lockfile, capture script; CI 3.11+3.12 | No | No | No |
| Release metadata | Owner-only | Checklist verified; nothing invented | No | No | No |

---

## P0 — Frozen representation provenance

**Issue:** Station feature vector wastes a slot by duplicating `interval.soc_lower`.

**Evidence:** Source assignment `arrival = interval.soc_lower`; tests show physical `_arrival_soc` can differ from continuation lower while frozen features still copy the lower bound into index 6.

**Action:** `docs/FROZEN_METHOD_NOTES.md`; `tests/rl/test_frozen_station_features.py`.

**Behavior / TEST:** unchanged.

---

## P1 — Presentation & reproducibility

**Actions:**

- Paper-facing p-values via `fmt_p` → `<5e-5` at Monte Carlo floor (`n_perm=20000`).
- Amount-sensitivity table note emphasizes envelope, not continuous superiority.
- Ablation labels: Base HPPO (B0), + Time-aware cap (B1), + Return scaling (B3), FA-HPPO full (B2).
- Environment docs + `requirements-experiment-lock.txt` (verified versions only) + `scripts/repro/capture_environment.py`.
- CI matrix adds Python 3.12 alongside 3.11.

---

## P2 — Strong comparator study

**Action:** `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md`.

**Recommendation:** do not insert a new baseline into consumed V3 confirmatory family; prefer **new predeclared V4 TEST** if a same-problem optimizer is established.

---

## P3 — Envelope / lower-bound development ablation

**Action:** prepared (not claimed complete):

- `scripts/development/run_envelope_ablation.py`
- `results/development/envelope_ablation/PROTOCOL.md`

Uses existing `AblationConfig(soc_interval=..., time_aware=...)` on SynthCharge TRAIN/VAL with return scaling on. Labeled **POST-HOC DEVELOPMENT**.

**Execution status:** infrastructure only in this repair commit. Full five-seed×three-variant CPU training is a long job; run explicitly when compute is available. Results must never enter V3 confirmatory tables.

---

## P4 — Frozen-raw failure analysis

**Action:** `tableA03_failure_routes.*`, `figA05_failure_consistency.png`.

Pre-failure trajectories: **unavailable** in V3 raw (not reconstructed; TEST replay forbidden).

---

## P5 — V4 candidates & title

**Action:** `docs/V4_METHOD_CANDIDATES.md` including feature fix, endpoint-capable continuous heads, stronger comparator, OOD/nonlinear benchmark, and title recommendation (prefer “Feasibility-Aware PPO…” for manuscript emphasis; keep FA-HPPO name in body; no mass rename).

---

## Verification checklist

1. Full pytest — run at end of repair  
2. CI — 3.11 + 3.12 unit-tests matrix  
3. Method-freeze diff empty vs `df35705` for `src`, `third_party`, listed configs  
4. V3 raw TEST hash unchanged vs `EVALUATION_CONSUMED`  
5. `TEST_LOCK` unchanged  
6. Checkpoint hashes unchanged  
7. `paired_primary.json` unchanged  
8. Consumed TEST not rerun  
9. New analyses labeled confirmatory vs development/post-hoc  
10. `results_paper/` regenerated without TEST evaluation  

---

## Remaining V4 recommendations

1. Correct distinct arrival vs continuation-lower station features; retrain.  
2. Consider endpoint-capable amount parameterizations (hypothesis, not V3 causal claim).  
3. Same-problem planning comparator under a **new** predeclared TEST.  
4. Complete SynthCharge TRAIN/VAL envelope ablation (A/B/C) as development evidence.  
5. Optional nonlinear / OOD benchmark — never overclaim SynthCharge linear energy as EVRPTW-GR physics.
