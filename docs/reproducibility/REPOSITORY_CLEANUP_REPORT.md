# Repository cleanup report

Date: 2026-10-06  
Scope: V4 Fig. 3 fix + major layout consolidation (no retraining, no TEST rerun).

## Summary counts

| Metric | Before | After |
|--------|--------|-------|
| Tracked/working files (approx., excl. `.git`) | 7963 | ~7965 |
| Top-level non-dot directories | 17 | 10 |
| Top-level scattered result dirs (`results_*`) | 3 | 0 |
| Top-level `checkpoints*` dirs | 5 | 0 |

## Final top-level tree

```text
RL_CHARGE/
├── README.md
├── RESULTS_INDEX.md
├── REPRODUCIBILITY.md
├── RELEASE_CHECKLIST.md
├── pyproject.toml
├── requirements-experiment-lock.txt
├── LICENSE.TEMPLATE
├── CITATION.cff.template
├── .github/
├── src/
├── tests/
├── configs/{common,v1,v2,v3,v4}/
├── data/
├── models/{v1,v2,v3,v4,development}/
├── results/{v1,v2,v3,v4,development,raw,runs,smoke}/
├── scripts/{common,v1,v2,v3,v4,paper,development,repro}/
├── paper/
├── docs/{methodology,reproducibility,archive}/
└── third_party/
```

## Folders removed (top-level)

- `checkpoints/`, `checkpoints_v2/`, `checkpoints_v3/`, `checkpoints_v4/`, `checkpoints_development/`
- `results_v4/`, `results_v4_paper/`, `results_paper/`
- Empty `paper/figures/`, `paper/tables/` placeholders
- Generated caches where removable (`.pytest_cache`, `__pycache__`; OneDrive may retain some locks)

## Folders created

- `models/{v1,v2,v3,v4,development}/`
- `results/v1/`, `results/v3/`, `results/v4/{paper,test,reward_development,training_figures}/`
- `scripts/{common,v1,v3,v4}/` (plus relocated `v4/reward`, `v4/test`)
- `configs/{common,v1,v3,v4}/`
- `docs/{methodology,reproducibility,archive}/`

## Fig. 3 fix

- Arrival SOC set to `0.25` with `SOC_lower=0.35` so `arrival ≤ lower`.
- Label: target lower bound = max(arrival SOC, energy-continuation requirement).
- Feasibility bar charts y-max set to 105% (was 110%).
- No scientific numbers changed.

## Model / checkpoint migration

| Old | New | Hash status |
|-----|-----|-------------|
| `checkpoints_v4/final_authoritative/V4_BASE_NO_L_FAIL/` | `models/v4/fa_hppo/` | SHA256 match vs `CHECKPOINT_FREEZE.json` (seeds 42–46) |
| `checkpoints_v4/reward_ablation/` | `models/v4/reward_ablation/` | moved intact |
| `checkpoints_v4/final_reward/` | `models/v4/final_reward/` | kept (hashes differ from authoritative; not deleted) |
| `checkpoints_v3/` | `models/v3/` | moved |
| `checkpoints_v2/` | `models/v2/` | moved |
| `checkpoints/` | `models/v1/` | moved |
| `checkpoints_development/` | `models/development/` | moved (OPTIONAL_ARCHIVE) |

Inventory: `docs/reproducibility/MODEL_INVENTORY.md`

## Result-folder migration

| Old | New |
|-----|-----|
| `results_v4_paper/` | `results/v4/paper/` |
| `results/v4_reward/` | `results/v4/reward_development/` |
| `results/v4_test/` | `results/v4/test/` |
| `results_v4/` | `results/v4/training_figures/` |
| `results_paper/` | `results/v3/paper/` |
| `results/v3_hppo/` | `results/v3/` |
| `results/final/` | `results/v1/final/` |
| `results/pilot/`, `results/summaries/` | `results/v1/pilot/`, `results/v1/summaries/` |
| `results/v2/` | unchanged under `results/v2/` |

## Markdown cleanup

- Active methodology docs → `docs/methodology/`
- Cleanup/status/history reports → `docs/archive/` (incl. former `docs/history/`, `context.md`)
- Root README / RESULTS_INDEX / REPRODUCIBILITY rewritten for new layout

## Scripts / configs

- Common utilities → `scripts/common/`
- V1 runners (incl. frvcpy) → `scripts/v1/`
- `scripts/v4_reward` → `scripts/v4/reward/`
- `scripts/v4_test` → `scripts/v4/test/`
- `scripts/v3_hppo` → `scripts/v3/`
- `configs/{rl,physics,experiments,routing}` → `configs/common/...`

## Scientific hash verification (unchanged)

| Artifact | SHA256 |
|----------|--------|
| raw TEST (LF) | `431014b397eba9191b09312f83ccb7523641e353f3d37ae9c6d57bb2a31b00dc` |
| TEST_LOCK (LF) | `c0055ba29beaa9b36590ef7e84cc9a60735f52adabaf7ca51dfdde5b89444d58` |
| EVALUATION_CONSUMED | `3cdcef0bb339f63a21466d5dd26d27e8c0d518820ed402de21eec78a02bc54b7` |
| FINAL_REWARD_FREEZE | `c6a520e1a588445c880056777c8298de5fbbf1ae1cdfeac3865b104e036dc82f` |
| V4 seed-42 `best.pt` | `759f8d093719b73e2910c6cebe7fec001b6c1cca638f75a50ce5478eae15c434` |

Frozen protocol JSON **contents** were not rewritten (historical path strings preserved). Resolver: `src/repo_paths.py` + `docs/reproducibility/PATH_MIGRATION.md`.

## Verification status

- `python -m pytest tests -q` → **256 passed**
- `python scripts/paper/build_v4_results_paper.py` → OK
- `python scripts/paper/build_v4_results_paper.py --verify` → **VERIFY OK**
- CI workflow still: Python 3.11 / 3.12 + frvcpy (`scripts/v1/run_frvcpy_benchmark.py`)

## Confirmations

- No training performed
- No TEST evaluation rerun
- No reward / SOC-envelope implementation / architecture changes
- Scientific result numbers unchanged (94.56% feasibility etc.)

## Intentionally kept (scientifically risky to delete)

- `models/v4/final_reward/` (distinct hashes from authoritative; historical training sidecars)
- `models/v4/reward_ablation/` and smoke trees
- `models/development/` envelope ablation models
- All V1/V2/V3 result trees
- Frozen manifests that still mention old paths (byte-identical provenance)
- `results/development/`, `results/runs/`, `results/smoke/`, `results/raw/`

## Final polish pass (post-`c46d3ec`)

- Tracked `models/README.md` + `models/v4/fa_hppo/CHECKSUMS.json`; `.gitignore` ignores only `models/**/*.pt`
- Removed one-shot migration helpers under `scripts/common/`
- Removed inaccurate `REPOSITORY_MIGRATION.json` / old cleanup-plan; rely on `PATH_MIGRATION.md` + this report
- Deleted redundant `docs/archive/` cleanup/history Markdown and `context.md`
- Slimmed `MODEL_INVENTORY.md` (JSON remains canonical)
- Removed duplicate `results/v4/training_figures/` (canonical training displays are paper figs 12–15 / A01)
- Moved `results|scripts|models/development` → `*/v3/development/`
- Moved V3 `paper/CLAIMS.md` + `RESULTS_MAP.md` → `results/v3/paper/claims/`

No training, no TEST rerun, frozen scientific hashes unchanged.
