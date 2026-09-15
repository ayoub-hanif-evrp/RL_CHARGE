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
   there exists an *energy-continuation* path `f → (stations)* → next frozen
   node` using full-battery hops on the same directed `EnergyModel` /
   `BatteryModel` as the simulator (intermediate recharges to max SOC allowed).
   Multiple hops stay legal. The loop-guard still applies. Same-station charges
   with `target_SOC − arrival_SOC < ε` are masked as `ZERO_CHARGE_NOOP`.
   Positive same-station charge remains legal.
3. If a station is selected, the FULL transition-level SOC interval is
   `[max(soc_on_arrival, continuation_soc_lower), max_soc]`.
   `continuation_soc_lower` is the minimum **departure** SOC at `f` that can
   take at least one first hop onto that continuation graph, including the
   battery reserve (`min_soc_fraction`), inverted through the battery model
   (not a naive `E/Q` if regeneration is present). Ablation A2 uses physical
   validity only: `[soc_on_arrival, max_soc]`. Map `u ∈ [0, 1]` by
   `target_soc = soc_lower + u * (soc_upper - soc_lower)`.
4. If every discrete bit is false, the episode is a terminal failure with
   `r = -(H - t)` where `H` is the depot due date. Failure `r = -(H − t)` is
   documented and is **not** the sole feasibility mechanism (model selection
   is lexicographic on validation feasibility, then completion time).

This continuation bound **is not globally exact**: it ignores time windows and
charging duration. Hard TW checks stay in the simulator.

Infeasible actions are masked, not sampled.
