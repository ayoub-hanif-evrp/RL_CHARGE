# RL_CHARGE — Feasibility-Aware Hybrid PPO for Fixed-Route EV Charging

Learn **when / where / how much** to charge on a **frozen** customer sequence.
The controller does **not** choose customer order.

## Layout (versioned)

| Kind | Path |
|------|------|
| Models | `models/v{1,2,3,4}/`, `models/development/` |
| Results | `results/v{1,2,3,4}/` |
| Configs | `configs/common/`, `configs/v{1,2,3,4}/` |
| Scripts | `scripts/v{1,2,3,4}/`, `scripts/paper/`, `scripts/common/` |

Legacy path aliases: `src/repo_paths.py`. Migration map: `docs/reproducibility/PATH_MIGRATION.md`.

## Current paper-facing version: V4

**FA-HPPO with normalized time-horizon reward**

- discrete head: CONTINUE or select a station;
- continuous amount \(u\in[0,1]\) mapped into
  \(\mathrm{SOC}_{target}=\mathrm{SOC}_{lower}+u\,(\mathrm{SOC}_{upper}-\mathrm{SOC}_{lower})\);
- reward \(r_t=-\Delta t/C_{\mathrm{train}}\) (feasible) and
  \(r_{\mathrm{fail}}=-(H-t)/C_{\mathrm{train}}\) (failure), with \(C_{\mathrm{train}}=10\) on SynthCharge;
- fresh independently generated held-out SynthCharge TEST (180 routes).

Standalone manuscript package:

```bash
python scripts/paper/build_v4_results_paper.py
python scripts/paper/build_v4_results_paper.py --verify
```

Outputs: `results/v4/paper/` (PNG figures + tables). Claims: `paper/V4_CLAIMS.md`.
Authoritative checkpoints: `models/v4/fa_hppo/`.

## Repository stages (independent)

| Stage | Location | Role |
|-------|----------|------|
| V1 | `results/v1/final/`, `models/v1/` | Historical archive |
| V2 | `results/v2/final/`, `models/v2/` | Historical archive |
| V3-HPPO | `results/v3/`, `models/v3/` | Historical confirmatory package |
| **V4** | `results/v4/{reward_development,test,paper}/`, `models/v4/` | **Current standalone paper package** |

Each version is independent for manuscript writing. Do not mix prior-version
comparisons into the V4 paper-facing package.

## Install / tests

```bash
pip install -e .
python -m pytest tests -q
```

## Immutable rules

- Do not retune from TEST.
- Do not modify frozen historical result trees.
- Do not reopen a consumed TEST.
- Certificate timeout ≠ infeasible.

Full commands: `REPRODUCIBILITY.md`. Index: `RESULTS_INDEX.md`.
