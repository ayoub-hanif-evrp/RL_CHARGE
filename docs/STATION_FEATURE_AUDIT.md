# Station feature audit (V3 frozen)

**Method freeze:** `df35705c012b4ce88674a5b8906176c97838b06c`  
**File:** `src/rl/features.py` → `_station_features`  
**Related:** `docs/FROZEN_METHOD_NOTES.md`, `tests/rl/test_frozen_station_features.py`

## Frozen 12-dimensional station vector

| Index | Variable / expression (as coded) | Intended reading | Frozen semantics |
|------:|----------------------------------|------------------|------------------|
| 0 | `arc_cf.distance` | current→station distance | geometric |
| 1 | `arc_cf.travel_time` | current→station travel time | geometric |
| 2 | `energy_cf.net_energy` | current→station energy | energy |
| 3 | `arc_fn.distance` | station→next distance | geometric |
| 4 | `arc_fn.travel_time` | station→next travel time | geometric |
| 5 | `energy_fn.net_energy` | station→next energy | energy |
| 6 | `arrival` where `arrival = interval.soc_lower` | name suggests arrival SOC | **duplicate of index 9** |
| 7 | `station.altitude` | altitude | terrain channel |
| 8 | detour time `t_cf+t_fn-t_cn` | detour | time |
| 9 | `interval.soc_lower` | feasibility lower departure SOC | continuation/arrival floor |
| 10 | `interval.soc_upper` | feasibility upper SOC | max / time-capped |
| 11 | slack after travel to station | next-node slack | time window |

## Is the duplication intentional?

**Conclusion: almost certainly accidental (naming bug), not a designed feature.**

Evidence:

1. **Introduction commit** `79438ad` (2026-09-15) introduced `_station_features` with `arrival = interval.soc_lower` already present. No later commit changed this assignment.
2. **Variable naming** uses `arrival` but assigns the interval lower bound, not `_arrival_soc(...)`.
3. **No documentation or tests** prior to the scientific-repair pass claimed that indices 6 and 9 should be identical by design.
4. **Mathematical distinction exists** in the shield: `_arrival_soc` (physical SOC upon reaching the station) can be strictly less than the energy-continuation lower departure SOC used in `continuation_to_max`.
5. Therefore the most plausible story is: an author intended index 6 to be arrival SOC, but wired it to `interval.soc_lower`.

This intent is **inferred**, not proven by a contemporaneous design note.

## What “actual arrival SOC” should be

\[
\text{actual\_arrival\_soc} = \_arrival\_soc(\text{sim}, \text{station})
\]

i.e. SOC after traveling `current_node → station` under the frozen energy model (or current SOC if already at the station).

Under `continuation_to_max`:

\[
\text{soc\_lower} = \max(\text{actual\_arrival\_soc},\;\text{energy-continuation bound}).
\]

So arrival and lower bound coincide only when the continuation bound does not raise the floor.

## Why this does not invalidate V3

Train and eval used the same duplicated representation. See `docs/FROZEN_METHOD_NOTES.md`.

## V4 correction (not implemented)

Distinct features:

- actual arrival SOC;
- energy-continuation / feasibility lower departure SOC;
- (time-aware) upper SOC.

Requires new method id, retrain, fresh TEST. See `docs/V4_PROPOSAL.md`.
