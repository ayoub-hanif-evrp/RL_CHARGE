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

The fleet is unrestricted (`num_available = n_customers` is a cap, not a
target). Vehicle `fixed_cost` is the dominating lexicographic penalty

```
F = 2 * n_customers * Dmax + 1
```

with `unit_distance_cost = 1`. One fewer vehicle always beats any distance
difference. Provenance field: `fleet_policy = unrestricted_fleet_with_fixed_cost`.

Stopping criterion: `MaxIterations` (not wall-clock). Defaults:

- Small_Network: 10_000
- Medium_Network: 20_000
- Large_Network: 40_000
- seed 42

## Terrain reuse

For each `(network_group, customer_folder, base_instance)` sibling set, customer
matrices are checked for identity. PyVRP is solved **once** on a canonical
sibling (`L` if present, else `NL`, else `VG`). Exact `customer_ids` tuples are
copied onto the other terrains with `route_source_instance_id` and
`terrain_reuse = true`. Terrain experiments then differ **only** in
altitude/energy.

## Serialization

Tracked in Git (frozen after Part 3A audit):

- `data/routes/corpus.jsonl`
- `data/routes/manifest.csv`
- `data/routes/corpus_metadata.json`

`data/routes/by_instance/` remains gitignored.

Each `FrozenRoute` records solver version, seed, iteration budget, config hash,
`F`, `n_vehicles`, unscaled `total_distance`, `lexicographic_objective`,
ordered customer IDs, and `charging_feasibility_status = "unverified"`.

No charging-station visits are stored. No route is dropped because a future
RL agent or heuristic could not charge it. Unassigned customers, if any, are
listed.

```bash
python scripts/generate_routes.py --profile official_evrptwgr --seed 42 --out data/routes
python scripts/audit_corpus.py --routes data/routes
python scripts/simulate_route.py --route data/routes/by_instance/c101C5_L.jsonl --policy continue
```

Continue-only simulation is a logger, not a filter. Customer TW replay with
infinite battery is stored as `routing_tw_feasible`.
