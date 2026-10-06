# RL_CHARGE — Feasibility-Aware Hybrid PPO for Fixed-Route EV Charging

Learn **when / where / how much** to charge on a **frozen** customer sequence.
The controller does **not** choose customer order.

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

Outputs: `results_v4_paper/` (PNG figures + tables). Claims: `paper/V4_CLAIMS.md`.

## Repository stages (independent)

| Stage | Location | Role |
|-------|----------|------|
| V1 | `results/final/` | Historical archive |
| V2 | `results/v2/final/` | Historical archive |
| V3-HPPO | `results/v3_hppo/` | Historical confirmatory package |
| **V4** | `results/v4_reward/`, `results/v4_test/`, `results_v4_paper/` | **Current standalone paper package** |

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
