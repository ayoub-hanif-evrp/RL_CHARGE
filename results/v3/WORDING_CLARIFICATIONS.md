# V3 wording and chronology clarifications

Locked JSON role strings in `TEST_LOCK.json` may still contain the historical token
`fresh_external_confirmatory_*`. That token is **not** a claim of distribution-shift
generalization to a non-SynthCharge domain.

## Preferred manuscript wording

- **fresh independently generated SynthCharge TEST**
- **fresh held-out SynthCharge confirmatory benchmark**
- stratified, certificate-filtered stress benchmark (180/2846 ≈ 6.3% candidate acceptance)

Not: “external generalization.”

## Chronology vs `EVALUATION_CONSUMED.json`

Accurate claim:

> No final FA-HPPO checkpoint was retrained, retuned, or reselected after TEST.

Development B0–B3 gold TRAIN/VAL ablations may finish after the TEST evaluation timestamp.
They were predeclared before TEST, do not touch TEST routes, and cannot change the
consumed TEST raw rows.

See updated fields in `EVALUATION_CONSUMED.json`.

## DiscretePPO history

V2 (`results/v2/final/`) retains HybridPPO vs DiscretePPO. V3’s predeclared confirmatory
matrix is FA-HPPO vs three heuristics only. Disclose V2 DPPO as historical evidence
outside the V3 comparison — do not erase or invalidate it.
