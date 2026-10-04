# Scientific fixes report

**Current HEAD (at authoring):** `72dac4b` + this fix pack  
**Method freeze:** `df35705c012b4ce88674a5b8906176c97838b06c`  
**Date context:** final scientific-quality pass after envelope ablation completion

---

## 1. Current HEAD / integrity

| Check | Status |
|-------|--------|
| Method-freeze diff (`src`, `third_party`, listed configs) vs `df35705` | empty |
| V3 raw TEST hash vs `EVALUATION_CONSUMED` | match |
| `TEST_LOCK` | 0 mismatches |
| Checkpoint SHA256 (5 FA-HPPO seeds) | all match freeze |
| `paired_primary.json` | untouched |
| Consumed TEST rerun | **no** |
| Full pytest | run at end of this pack |
| CI | matrix Python 3.11 + 3.12 (`tests.yml`); `gh` CLI not available locally |

---

## 2. V3 integrity status

**Preserved.** No checkpoint retrain/reselection, no TEST membership change, no raw rewrite, no silent `src/` contamination.

---

## 3. Station-feature audit conclusion

See `docs/STATION_FEATURE_AUDIT.md`.

- Indices **6 and 9 are duplicated** (`arrival = interval.soc_lower`).
- Introduced in commit `79438ad`; never corrected.
- **Almost certainly accidental** (name says arrival; value is interval lower).
- Does **not** invalidate V3 (train=eval representation).
- Correction → **V4-A** only.

---

## 4. Tests added

| Test file | Role |
|-----------|------|
| `tests/rl/test_frozen_station_features.py` | Provenance: freeze duplicate + arrival≠continuation diagnostic |
| `tests/rl/test_semantic_soc_and_training_invariants.py` | \(SOC_{tgt}\) mapping; arrival≠lower; time-aware tighten; normalizer TRAIN-only; return_scale>0 |

---

## 5. Beta-head analysis

See `docs/CONTINUOUS_HEAD_LIMITATION.md`.  
α,β>1; eval uses Beta mean; Max≈free motivates V4 endpoint-capable heads — **not** a causal V3 bug claim.

---

## 6. Lower-bound ablation status

**Completed** (SynthCharge TRAIN/VAL, seeds 42–46):

| Variant | Mean VAL feas. |
|---------|----------------|
| Arrival-to-Max | 0.993 |
| EnergyLower-to-Max | 0.989 |
| EnergyLower-to-TimeUpper | 0.982 |

Path: `results/development/envelope_ablation/`  
Label: **POST-HOC / DEVELOPMENT-ONLY MECHANISM STUDY — NO V3 TEST**.

Honest note: Arrival-to-Max is slightly best on this VAL set; do not rewrite V3 confirmatory narrative from this.

Missing from initial validation schema (documented): per-variant NO_FEASIBLE_ACTION / visits / charging-time histograms (would need optional re-rollout of development checkpoints).

---

## 7. Same-domain SynthCharge B0/B1/B3/B2 status

**Script + protocol prepared, not fully executed** in this pack.

- `scripts/development/run_synthcharge_b_ablation.py`
- Estimate: ~15–22 CPU-hours (20 cells)

Gold B0–B2 remains the paper development ablation. SynthCharge 2×2 is post-hoc robustness if/when run.

---

## 8. Optimization comparator feasibility study

- `docs/OPTIMIZATION_COMPARATOR_DESIGN.md`
- `docs/STRONG_COMPARATOR_FEASIBILITY_STUDY.md`

Prefer TRAIN/VAL prototypes; confirmatory comparison → new V4 TEST. frvcpy is not a V3 oracle.

---

## 9. Failure-analysis findings

`results_paper/failure_analysis/` (+ tableA03 / figA05)

- 52 route×seed failures; reason **NO_FEASIBLE_ACTION** only in frozen rows.
- **20** routes with ≥1 failing seed; **5** routes fail for **all five** seeds.
- Seed 46 has more failures than seed 45 (see FINDINGS.json).
- Pre-failure trajectories: **unavailable** without forbidden TEST replay.

---

## 10. Environment-lock changes

- `docs/ENVIRONMENT_REPRODUCIBILITY.md`
- `requirements-experiment-lock.txt` (verified versions only; not full freeze)
- `scripts/repro/capture_environment.py`
- CI Python 3.11 + 3.12

Exact Torch 2.14.0 on Ubuntu CI may differ from Windows experiment wheels — documented, not claimed bit-identical.

---

## 11. CI status

Workflow updated for 3.11/3.12. Local `gh` unavailable; rely on GitHub Actions after push.

---

## 12. Paper / documentation changes

- Claim discipline reinforced (`paper/CLAIMS.md`): envelope-centric; Max≈free honest.
- Named audit docs: station feature, continuous head, comparator, V4 proposal.
- Release templates: `LICENSE.TEMPLATE`, `CITATION.cff.template` (owner TODOs).
- `results_paper/` remains single publication-output directory.

---

## 13. V4 recommendations

See `docs/V4_PROPOSAL.md` (V4-A…E). **Do not train without approval.**

---

## 14. Separation of evidence

### FIXED NOW WITHOUT CHANGING SCIENCE
- Documentation of frozen quirks
- Provenance/semantic tests
- Paper-facing p-value floor formatting
- Failure analysis from frozen raw
- Env/CI reproducibility layer
- Release templates (no invented metadata)

### NEW DEVELOPMENT-ONLY EVIDENCE
- SynthCharge envelope A/B/C ablation (completed)
- Failure concentration report
- (Optional) SynthCharge B0–B2 2×2 when compute available

### REQUIRES V4 / NEW TEST
- Correct arrival feature
- Endpoint-capable continuous head
- Strong planner in confirmatory family
- Fresh TEST / optional nonlinear OOD
