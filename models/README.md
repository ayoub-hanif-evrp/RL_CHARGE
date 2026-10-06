# Trained models

Weight binaries (`.pt`) are **not** stored in ordinary Git history.

They live locally under this tree (or are published separately via GitHub Releases / Zenodo / Git LFS):

```text
models/
├── v1/                 # historical V1
├── v2/                 # historical V2
├── v3/
│   └── development/    # V3 post-hoc envelope / SynthCharge-B ablations
└── v4/
    ├── fa_hppo/        # authoritative V4 FA-HPPO (seeds 42–46) — paper TEST
    ├── reward_ablation/
    └── final_reward/   # earlier clean run sidecars (distinct hashes)
```

## Authoritative V4 checkpoints

Canonical path: `models/v4/fa_hppo/seed_{42..46}/best.pt`

SHA256 checksums: [`v4/fa_hppo/CHECKSUMS.json`](v4/fa_hppo/CHECKSUMS.json)  
These match `results/v4/test/CHECKPOINT_FREEZE.json` and `results/v4/reward_development/FINAL_REWARD_FREEZE.json`.

## Full inventory

Machine-readable inventory (all local `.pt` files when present):  
[`docs/reproducibility/MODEL_INVENTORY.json`](../docs/reproducibility/MODEL_INVENTORY.json)

## Obtaining binaries

1. Prefer local copies produced by the frozen training runs (hashes above).
2. Otherwise restore from the archival release / Zenodo bundle for this paper version.
3. Never retrain to “fill” this folder for reproducibility of the locked TEST.

Path aliases for older docs: see `docs/reproducibility/PATH_MIGRATION.md` and `src/repo_paths.py`.
