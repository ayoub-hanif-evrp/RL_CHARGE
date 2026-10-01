# FINAL V2 EXPERIMENT REPORT

This report distinguishes archived V1, V2 development evidence, frozen B2 methodology,
final gold training, the fresh SynthCharge TEST, and the non-fresh legacy EVRPTW-GR challenge.

## Provenance

- method freeze SHA: `df35705c012b4ce88674a5b8906176c97838b06c`
- training execution SHA: `2f4825adc341b71cec43e208540ee383de29cfb4`
- evaluation/analysis git HEAD: `8179c2dfa7c84ac9aebf61d12ba8e6913991551a`
- CHECKPOINT_FREEZE sha256 (LF): `07006cccb80c696aeec4aa1c4899724abc743c2075795ad0f2110964d8aac12a`
- TEST_LOCK sha256 (LF): `6a37a654be34cb8cea82db9d80b6417f65692082ed8c6c7915a350dc01609c1f`
- environment: Python 3.12.10, Torch CPU, PyVRP 0.14.0, frvcpy 0.1.1

## Method

Proposed method: **Hybrid-Action PPO (HybridPPO)**, development condition **B2**:
energy-continuation SOC lower bound + optimistic time-feasibility SOC upper bound +
one global TRAIN-only return scale.

Learned comparator: DiscretePPO (same shield, same return scale, same budget; discrete charge amounts).
Excluded from the final matrix: DDQN, SAC, AttentionPPO.

Reward documentation (unchanged): successful training return = `-T_completion`;
failed training return = `-(H - t0) - L_remaining`. Evaluation infeasible completion = `H`.

## Development evidence (not TEST)

Pre-freeze B0/B1/B2/B3 and B2_pyvrp diagnostics remain under `results/v2/diagnostics/`.
They used TRAIN/VAL only and motivated freezing B2. They are not fresh TEST results.

## Fresh SynthCharge TEST (90 routes, linear physics)

Physics profile: `synthcharge_linear` (`energy_law=linear_distance`).
This does **not** validate EVRPTW-GR gradient physics.

- HybridPPO feasibility (mean±SD across 5 seeds): 0.902 ± 0.027
  95% hierarchical bootstrap CI: [0.867, 0.936]
  per-seed: {42: 0.8888888888888888, 43: 0.9222222222222223, 44: 0.9, 45: 0.9333333333333333, 46: 0.8666666666666667}
- DiscretePPO feasibility: 0.907 ± 0.015
  95% CI: [0.878, 0.936]
  per-seed: {42: 0.9, 43: 0.9, 44: 0.9222222222222223, 45: 0.8888888888888888, 46: 0.9222222222222223}
- HybridPPO failure-retaining completion: 4.395 ± 0.159
  95% CI: [4.173, 4.626]
- DiscretePPO failure-retaining completion: 4.279 ± 0.086
  95% CI: [4.088, 4.473]

### Charging-required subset (72 routes)

- HybridPPO feasibility: 0.881 ± 0.032
- DiscretePPO feasibility: 0.883 ± 0.019

### Primary paired comparison (HybridPPO − DiscretePPO)

- feasibility: effect=-0.0044, 95% CI=[-0.0267, 0.0156], p_raw=0.8505, p_Holm=0.8505, n=90
- completion_all_routes: effect=0.1158, 95% CI=[-0.0035, 0.2474], p_raw=0.06355, p_Holm=0.1271, n=90

Do not claim statistical superiority unless Holm-adjusted evidence supports it.

## Legacy same-domain challenge (26 routes, NOT a fresh TEST)

Gold-trained checkpoints only. Six historically consumed V1 TEST parents.
Not used for model selection, tuning, or fresh-generalization claims.

- HybridPPO feasibility: 0.815 ± 0.050
- DiscretePPO feasibility: 0.915 ± 0.050

## Limitations

- SynthCharge uses linear energy; no EVRPTW-GR terrain/payload energy coupling.
- Certificate search is a positive witness generator; timeout/exhaustion is unverified, not infeasible.
- Legacy challenge has only six parents; low statistical power.
- Five training seeds; seed variability is reported and must not be pooled as independent routes.
- Stateless baselines are secondary references under the same V2 shield where applicable.

## Artifacts

- raw: `results/v2/final/raw/`
- statistics: `results/v2/final/statistics/`
- tables: `results/v2/final/tables/`
- figures: `results/v2/final/figures/`
- checkpoint freeze: `results/v2/final/CHECKPOINT_FREEZE.json`

