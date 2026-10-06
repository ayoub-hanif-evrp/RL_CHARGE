# Cleanup Manifest

Starting HEAD: `557f1b38228d6d44fca34331805ecfde4f3d2fed`  
Audit date: 2026-10-02  
Tests at audit: **226 passed**

## Policy

- Prefer **preservation + documentation** over destructive moves.
- **Never modify** `results/v2/final/` (historical V2 HybridPPO/DiscretePPO evidence).
- **Never modify** `results/final/` (archived V1).
- **Never modify** `data/routes/`, `data/splits/`, frozen `data/routes_v2/`, `data/splits_v2/` for aesthetics.
- Soft cleanup only: documentation, paper namespace, ignore caches, root README.

## Inventory summary

| Area | Role | Action |
|------|------|--------|
| `results/v2/final/` | Frozen V2 confirmatory experiment (includes DiscretePPO) | **KEEP untouched** |
| `results/final/` | Archived V1 paper results | **KEEP**; document as historical |
| `results/pilot/`, `results/smoke/`, `results/raw/` | Development / pilot | **KEEP**; document as non-paper |
| `checkpoints/`, `checkpoints_v2/` | Local binaries (gitignored) | **KEEP** hashes in freeze manifests |
| `third_party/SynthCharge_v1.0/` | Vendored generator + LICENSE | **KEEP** |
| `.ipynb_checkpoints/`, `.pytest_cache/` | Local caches | Already gitignored; not in git |
| `results/final/figures/exact_small/placeholder.png` | V1 placeholder figure | **KEEP** (historical V1); not used by V3 paper |
| `scripts/*.py` (root) | V1 paper pipeline | **KEEP**; document as V1 archive tooling |
| `scripts/v2/` | V2 finalization | **KEEP** |
| `context.md` | Lab handoff | Update status; do not delete |

## Items considered for removal — REJECTED

| Path | Reason kept |
|------|-------------|
| `results/v2/final/**` | Immutable V2 evidence including DiscretePPO |
| `results/final/**` | Immutable V1 evidence |
| Root `scripts/*.py` | Needed to regenerate archived V1 tables/figures |
| `docs/*.md` | Still describe frozen simulator/physics |

## Soft actions taken (this workstream)

| Old path | New path | Reason | Referenced? | Recoverable? | Affects reproducibility? |
|----------|----------|--------|-------------|--------------|--------------------------|
| *(none deleted)* | — | — | — | — | — |
| — | `docs/CLEANUP_MANIFEST.md` | Audit record | n/a | yes | no |
| — | `paper/` | Paper-facing map | n/a | yes | no |
| — | `results/v3_hppo/` | New HPPO-only confirmatory namespace | n/a | yes | adds new evidence |
| — | `results/v3_hppo/configs/` | Paper TEST generator config (kept out of frozen `configs/` tree) | n/a | yes | no |
| — | `README.md`, `REPRODUCIBILITY.md`, `RESULTS_INDEX.md`, `RELEASE_CHECKLIST.md` | Reviewer entry points | n/a | yes | no |
| — | `docs/ARCHIVE_LAYOUT.md` | Explains V1/V2/V3 roles without moving files | n/a | yes | no |
| — | `scripts/v3_hppo/` | Generation / freeze / eval / ablation tooling | n/a | yes | no |
| — | `paper/` | Claims, results map, figures/tables | n/a | yes | no |

## Seed-range audit (SynthCharge)

Used generator seeds in existing SynthCharge artifacts: **100000–300550**.  
**400000+ is unused** (confirmed against split manifests and `candidate_audit.jsonl`).  
V3 confirmatory TEST will use seed start **400000**.

## Method freeze invariant

Scientific algorithm/physics freeze (must stay empty):

`git diff df35705..HEAD -- src third_party configs/rl configs/physics configs/routing configs/experiments configs/v2`

New V3 protocol tests under `tests/test_v3_*` are allowed. V3 work otherwise lives under `scripts/`, `docs/`, `paper/`, `results/v3_hppo/` unless a stop condition is hit.
