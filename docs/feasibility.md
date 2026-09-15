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
2. **Station `f`** is unmasked iff travel `current → f` is energy-feasible and
   `f` is a station. The shield does **not** require `f → next customer`.
   Multiple station hops remain legal. The loop-guard still applies.
3. If a station is selected, the transition-level SOC interval is
   `[soc_on_arrival, max_soc]`. Map `u ∈ [0, 1]` by
   `target_soc = soc_lower + u * (soc_upper - soc_lower)`. This interval is
   **not** claimed globally exact.
4. If every discrete bit is false, the episode is a terminal failure with
   `r = -(H - t)` where `H` is the depot due date.

Infeasible actions are masked, not sampled.
