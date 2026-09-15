# Frozen-route corpus

Stage A of the methodology: generate customer sequences with a strong external
VRPTW solver, freeze them, and store provenance. Stage B (charging) never
edits these files.

## What is solved

PyVRP 0.14.0 solves a **customer-only capacitated pickup-VRPTW**. Charging
stations, battery, altitude, and energy are omitted on purpose.

Demand is mapped to PyVRP **pickup**, matching the EVRPTW-GR pickup convention
(load increases along the tour). Using delivery would reverse the payload
trajectory.

Stopping criterion: `MaxIterations` (not wall-clock). Defaults, chosen
independently of any charging method:

- Small_Network: 10_000
- Medium_Network: 20_000
- Large_Network: 40_000
- seed 42

Cross-platform bit-identity is not claimed. Terrain variants `_L` / `_NL` /
`_VG` of the same base instance produce identical customer matrices because
altitude is ignored.

## Serialization

Generated files live under `data/routes/` (gitignored except `.gitkeep`):

- `corpus.jsonl`
- `by_instance/{instance_id}.jsonl`
- `manifest.csv`
- `failures.json`

Each `FrozenRoute` records dataset identity, solver version, seed, iteration
budget, config hash, physics profile, integerization scales, instance SHA-256,
ordered customer IDs, demand, distance, routing feasibility, and
`charging_feasibility_status = "unverified"`.

No charging-station visits are stored. No route is dropped because a future
RL agent or heuristic could not charge it.

There is no train/validation/test split in Part 2.

## Commands

```bash
python scripts/generate_routes.py --profile official_evrptwgr --seed 42 --out data/routes
python scripts/simulate_route.py --route data/routes/by_instance/c101C5_L.jsonl --policy continue
```

Continue-only simulation is a logger, not a filter.
