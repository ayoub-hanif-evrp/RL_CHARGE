# Envelope lower-bound ablation protocol

**POST-HOC DEVELOPMENT / TRAIN–VAL ONLY — NOT V3 CONFIRMATORY TEST EVIDENCE**

## Purpose

The gold B0/B1/B3/B2 study varies time-aware upper bound × return scaling, but
keeps `soc_interval=continuation_to_max`. This development study isolates the
**energy-continuation lower SOC bound** on **SynthCharge TRAIN/VAL**.

## Variants

| Code | Display name | `soc_interval` | `time_aware` | Return scale |
|------|--------------|----------------|--------------|--------------|
| A_arrival_to_max | Arrival-to-Max | `arrival_to_max` | False | on |
| B_energy_to_max | EnergyLower-to-Max | `continuation_to_max` | False | on |
| C_energy_to_time | EnergyLower-to-TimeUpper | `continuation_to_max` | True | on |

Uses existing frozen `AblationConfig` fields only (no `src/` edits).

## Budget / seeds

- PPO config: `configs/common/rl/hybrid_ppo.toml` (same as V3 training budget)
- Seeds: 42–46
- Dataset: `data/routes_v2/synthcharge_final/{train,validation}`
- Checkpoint selection: TRAIN/VAL only (standard `train_hybrid_ppo`)

## Forbidden

- Reading or evaluating consumed V3 TEST
- Editing `paired_primary.json` or V3 confirmatory tables
- Claiming confirmatory status

## Launch

```bash
python scripts/development/run_envelope_ablation.py --variant all
# or one cell:
python scripts/development/run_envelope_ablation.py --variant A_arrival_to_max --seed 42
```

Outputs: `results/development/envelope_ablation/{variant}/seed_{s}/validation.json`
Checkpoints: `models/development/envelope_ablation/...`
