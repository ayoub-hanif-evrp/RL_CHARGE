# Feasibility shield

The shield is method-independent. It **masks** illegal simulator transitions. It
does **not** choose when to charge.

Three labels stay separate:

| Label | Meaning |
| --- | --- |
| `transition_feasible` | The simulator accepted the last `ContinueAction` / `ChargeAction`. |
| `shield_mask` | Which discrete actions are legal now. |
| `charging_feasibility_status` | Global existence of a charging schedule. Stays `unverified` in Part 3A. |

## Mask

At state `s`:

1. **CONTINUE** is unmasked iff the next frozen node (customer or depot) is a
   legal simulator `ContinueAction`: energy, customer/depot time window
   (`service_start ≤ due_date`), capacity after the load convention, known node.
2. **Station `f`** is unmasked iff travel `current → f` is energy-feasible **and**
   there exists an *energy-continuation* path `f → (unvisited stations)* →
   next frozen node` using full-battery hops on the same directed
   `EnergyModel` / `BatteryModel` as the simulator (intermediate recharges
   to max SOC allowed). Stations already visited since the last customer
   progress event are excluded from the continuation graph, so a candidate
   cannot be certified through a now-illegal hop. Revisit cycles
   (`S1 → S1`, `S1 → S2 → S1`) are masked as `STATION_REVISIT`. The
   programming loop-guard remains a backstop. Same-station charges with
   `target_SOC − arrival_SOC < ε` are still rejected as `ZERO_CHARGE_NOOP`.
   After the next frozen customer is served, previously used stations may be
   selected again.
3. If a station is selected, the FULL transition-level SOC interval is
   `[max(soc_on_arrival, continuation_soc_lower), max_soc]`.
   `continuation_soc_lower` is the minimum **departure** SOC at `f` that can
   take at least one first hop onto that continuation graph, including the
   battery reserve (`min_soc_fraction`), inverted through the battery model
   (not a naive `E/Q` if regeneration is present). Ablation A2 uses physical
   validity only: `[soc_on_arrival, max_soc]`. Map `u ∈ [0, 1]` by
   `target_soc = soc_lower + u * (soc_upper - soc_lower)`.
4. If every discrete bit is false, the episode is a terminal failure with
   `r_fail = -(H - t0) - L_remaining(failure_state)`, where `t0` is the
   pre-action decision time, `H` is the depot due date, and `L_remaining`
   is a frozen-route travel+service lower bound (charging and waiting
   omitted). This is a horizon-derived failure-progress term, not a second
   operational objective. Model selection is lexicographic on
   **parent-balanced** validation feasibility, then parent-balanced
   completion time. Route-weighted VAL metrics are logged but do not select
   checkpoints.

This continuation bound **is not globally exact**: it ignores time windows and
charging duration. Hard TW checks stay in the simulator.

Infeasible actions are masked, not sampled.
