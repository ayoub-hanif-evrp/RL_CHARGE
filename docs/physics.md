# Physical model

This project studies **fixed-route electric-vehicle charging**. Customer routing
and charging are separate stages:

1. An external VRPTW solver (PyVRP) produces a customer sequence.
2. That sequence is frozen. Charging may insert station visits but must never
   reorder customers.

This document describes the EVRPTW-GR physical energy model implemented in
`src/physics`. Reinforcement learning, frvcpy, nonlinear charging curves, and
train/validation/test splits belong to Part 3.

## Units and sign conventions

| Quantity | Type | Meaning |
| --- | --- | --- |
| Distance | `Distance` | Planar Euclidean `hypot(Δx, Δy)` in instance coordinate units. **Not** the 3-D slope length. |
| Travel time | `TravelTime` | `distance / v` with file `v`. In every published file `v = 1`, so time equals distance in Schneider units. |
| Energy | `Energy` | Normalized Schneider units, the same space as battery `Q`. Positive = consumption, negative = regeneration before the SOC ceiling. |
| Battery energy | `BatteryEnergy` | Stored energy in Schneider units. `battery_after = battery_before - net_arc_energy`. |
| SOC | `SocFraction` | `battery_energy / Q`. Benchmark bounds: `[0, 1]`. |
| Payload | `PayloadMass` | Cargo kilograms. Pickup: starts at 0, increases by customer demand after service. Stations do not change payload. Gross mass = curb + payload. |
| Altitude | `Altitude` | File elevation in coordinate units, **not claimed to be SI metres**. Magnitudes are O(1). |
| Gradient | `Gradient` | `sin(arctan(Δh / d))`, matching Model 2. Percent slope is stored on the arc for audit only. |

These are distinct types. Subtracting a `Distance` from a `BatteryEnergy` is a
`TypeError`.

File `r` (consumption rate) is recorded and unused. Energy is never `r * distance`.

## Dual velocity

- **Time windows** use file `v` (always 1.0).
- **The kW formula** uses 60 km/h, then divides by that speed and by the
  empty-flat baseline. The 60 km/h parameter does not convert coordinates into
  travel time.

Working interpretation, not a dataset-PDF fact: if coordinates are kilometres
and time is minutes, `v = 1` km/min equals 60 km/h. The implementation does not
silently treat this as SI.

## Energy formula

Copied from EVRPTW-GR Model 2 linearized
(`sinarastani/EVRPTW-GR`, `EVRPTW-GR-Model-2-Linearized.py`) / Demir et al. (2012):

```
p_tract_kW = (F_roll + F_aero + F_grade + F_accel) * v_phys_m_s / 1000
p_batt_kW  = (p_tract / μ_train) / μ     if p_tract > 0
           = η_regen * p_tract           if p_tract <= 0
e_phys     = p_batt_kW / 60              # kWh per coordinate-km at 60 km/h
e_norm     = e_phys / e_phys_empty_flat  # empty-flat = 1
E_arc      = e_norm * distance           # Schneider units, comparable to Q
```

Default Demir constants (see `configs/physics/official_evrptwgr.toml`): curb
6350 kg, cargo cap 3650 kg, `cr=0.01`, `cd=0.7`, `rho=1.2041`, `A=3.912`,
`g=9.81`, `a=0`, `μ=μ_train=0.9`, `η_regen=0.8`, accessories 0.

The empty-flat baseline is approximately 0.370677 kWh per coordinate-km. `Q` is
**not** converted into kWh in the battery state. That figure is an audit
conversion only.

Energy is directional: `energy(i, j, payload)` need not equal
`energy(j, i, payload)`.

## Named profiles

- `official_evrptwgr` (**default**): curb 6350 kg and cargo 3650 kg from Model 2,
  not from Small_Network `C`/`M`. File `Q`, `g`, `v` still come from the instance.
  Cargo is enforced at 3650 kg. This is stricter than the published Model 2
  script, which uses `weight + sum(demand)` as a gross-weight bound.
- `raw_file`: curb from `M` and capacity from `C`. Missing `M` is an error, never
  a silent 6350.

Initial SOC defaults to 1 (thesis overnight full charge). The MILP does not
force `y_depot = Q`; the field is named so it can be changed later.

Regeneration is clipped at `max_soc * Q`. Surplus is discarded.

## Charging

`BenchmarkCompatibleLinearChargingModel`: `charging_time = g * energy_added`.

`GenericPiecewiseLinearChargingModel` requires explicit SOC and cumulative-time
knots. **No default nonlinear curve is invented.** EVRPTW-GR does not publish one.

Stations have location, altitude, and the instance-wide `g`. Price, queue wait,
and charger power are not in the dataset and are not defaulted.

## What this package does not do

Exact fixed-route charging feasibility (frvcpy), RL, nonlinear experimental
curves, SOC-reserve sensitivity experiments, and train/validation/test splits
are Part 3.
