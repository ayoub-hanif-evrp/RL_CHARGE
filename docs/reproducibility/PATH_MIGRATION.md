# Path migration map

Frozen protocol/manifest JSON files that embed historical paths are **byte-preserved**.
Use this map (and `src/repo_paths.py`) to resolve old paths to canonical locations.

| Old path | New canonical path |
|---|---|
| `checkpoints_v4/final_authoritative/V4_BASE_NO_L_FAIL/` | `models/v4/fa_hppo/` |
| `checkpoints_v4/reward_ablation/` | `models/v4/reward_ablation/` |
| `checkpoints_v3/` | `models/v3/` |
| `checkpoints_v2/` | `models/v2/` |
| `checkpoints/` | `models/v1/` |
| `checkpoints_development/` | `models/v3/development/` |
| `models/development/` | `models/v3/development/` |
| `results_v4_paper/` | `results/v4/paper/` |
| `results_v4/` / `results/v4/training_figures/` | `results/v4/paper/` (canonical displays: figs 12–15, A01) |
| `results/v4_reward/` | `results/v4/reward_development/` |
| `results/v4_test/` | `results/v4/test/` |
| `results_paper/` | `results/v3/paper/` |
| `results/v3_hppo/` | `results/v3/` |
| `results/development/` | `results/v3/development/` |
| `results/final/` | `results/v1/final/` |
| `scripts/v4_reward/` | `scripts/v4/reward/` |
| `scripts/v4_test/` | `scripts/v4/test/` |
| `scripts/v3_hppo/` | `scripts/v3/` |
| `scripts/development/` | `scripts/v3/development/` |
| `scripts/run_frvcpy_benchmark.py` | `scripts/v1/run_frvcpy_benchmark.py` |
| `paper/CLAIMS.md`, `paper/RESULTS_MAP.md` | `results/v3/paper/claims/` |

Weight binaries under `models/**/*.pt` are gitignored; tracked docs/checksums live in `models/`.
