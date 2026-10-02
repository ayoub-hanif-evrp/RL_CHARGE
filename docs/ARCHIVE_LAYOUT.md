# Archive layout (V1 / V2 / V3)

Paths are left in place for provenance. This document explains roles; it does **not** authorize moving frozen files.

## V1 — archived early work

| Path | Role |
|------|------|
| `results/final/` | Frozen V1 paper tables/figures/stats |
| `scripts/*.py` (repo root) | V1 regeneration tooling |
| `data/routes/`, `data/splits/` | V1 corpora |
| `checkpoints/` | V1 local binaries (gitignored) |

## V2 — historical HybridPPO + DiscretePPO freeze

| Path | Role |
|------|------|
| `results/v2/final/` | **Immutable** confirmatory freeze (includes DiscretePPO) |
| `results/v2/diagnostics/` | Two-seed B0–B3 development diagnostics |
| `data/routes_v2/`, `data/splits_v2/` | Frozen V2 corpora |
| `checkpoints_v2/final/` | Local B2 checkpoints (hashes in freeze) |
| `scripts/v2/` | V2 finalization tooling |

V2 is **valid historical evidence**. It is simply not the confirmatory experiment for the FA-HPPO-centered paper.

## V3-HPPO — paper confirmatory

| Path | Role |
|------|------|
| `results/v3_hppo/` | Protocol, locks, raw TEST rows, stats, ablation |
| `data/routes_v2/synthcharge_v3_test/` | Fresh untouched SynthCharge TEST (~180) |
| `data/splits_v2/synthcharge_v3_test.json` | Split manifest |
| `configs/v3_hppo/` | Paper TEST generator config |
| `scripts/v3_hppo/` | Generation / freeze / eval / ablation |
| `paper/` | Claims, results map, regenerated figures/tables |
| `scripts/paper/` | Artifact builder (no train / no TEST eval) |

## Soft-cleanup policy

No frozen V1/V2 scientific artifacts were deleted or relocated for aesthetics.
See `docs/CLEANUP_MANIFEST.md`.
