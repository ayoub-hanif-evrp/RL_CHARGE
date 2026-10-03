# RL_CHARGE — Feasibility-Aware Hybrid PPO for Fixed-Route EV Charging

Learn **when / where / how much** to charge on a **frozen** customer sequence.
The controller does **not** choose customer order.

## Proposed method

**Feasibility-Aware Hybrid PPO (FA-HPPO / HybridPPO, condition B2)**

- discrete head: CONTINUE or select a station;
- continuous amount \(u\in[0,1]\) mapped into
  \(\mathrm{SOC}_{target}=\mathrm{SOC}_{lower}+u\,(\mathrm{SOC}_{upper}-\mathrm{SOC}_{lower})\);
- \(\mathrm{SOC}_{lower}\): energy-continuation requirement;
- \(\mathrm{SOC}_{upper}\): optimistic time-feasibility cap;
- one global **TRAIN-only** return scale.

Generic Hybrid PPO is not claimed as novel. The contribution is
**feasibility-aware charging control** (envelope + scaling + HybridPPO
implementation), certified routes, and evaluation protocol. On the V3 TEST,
forcing \(u=1\) (FA-HPPO-Max) matches free amount control closely — see
`paper/CLAIMS.md`.

## Repository stages

| Stage | Location | Role |
|-------|----------|------|
| V1 | `results/final/` | Archived early experiment |
| V2 | `results/v2/final/` | Historical HybridPPO **and** DiscretePPO freeze (preserved) |
| V3-HPPO | `results/v3_hppo/` | Fresh independently generated SynthCharge confirmatory TEST: **FA-HPPO** + heuristics |

See `docs/ARCHIVE_LAYOUT.md`, `RESULTS_INDEX.md`, and `results/v3_hppo/WORDING_CLARIFICATIONS.md`.

## Install / tests

```bash
pip install -e .
python -m pytest tests -q
```

Pinned scientific freeze: method SHA `df35705c012b4ce88674a5b8906176c97838b06c`
(environment recorded in `results/v2/final/METHOD_FREEZE.json` /
`results/v3_hppo/ENVIRONMENT.json`).

## Regenerate publication figures/tables (no TEST rerun)

```bash
python scripts/paper/build_results_paper.py
python scripts/paper/build_results_paper.py --verify
```

Outputs: `results_paper/` (PNG figures + tables).

## Immutable rules

- Do not retune from TEST.
- Do not modify `results/v2/final/` or `results/final/`.
- Do not evaluate SynthCharge with `official_evrptwgr` physics.
- Certificate timeout ≠ infeasible.

Full commands: `REPRODUCIBILITY.md`.
