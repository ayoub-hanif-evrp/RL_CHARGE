# Fixed-route simulator

`FixedRouteSimulator` is a deterministic physics engine. It has no Gym, PyTorch,
or learned-policy dependency.

The customer sequence is an immutable tuple. Station visits may be inserted
between consecutive customers; the customer order cannot change.

## Actions

The caller chooses every decision. There is no rule-based "charging needed"
trigger and no six-level SOC grid.

- `ContinueAction`: travel to the next frozen customer, or to the depot if the
  sequence is finished. Applies load-dependent directional energy, waiting,
  service, then the required `LoadConvention` (pickup: `payload += demand`;
  delivery: `payload -= demand`). Advances the customer index.
- `ChargeAction(station_id, target_soc)`: travel to that station and charge
  continuously to `target_soc ∈ [current_soc, max_soc]`. Does **not** advance
  the customer index. Distinct stations may be used between two frozen
  customers; a station already visited since the last customer/depot progress
  event is `STATION_REVISIT` (same-station drip charging and `S1→S2→S1`
  cycles). After the next frozen customer is served, those stations may be
  used again.

A programming-loop guard (`loop_guard_station_visits`, default 50) remains a
defensive backstop against infinite charge cycles. It is not a scientific
"maximum three stops" constraint. True zero-ΔSOC charges at the current
station are still `ZERO_CHARGE_NOOP`.

## Time windows

Canonical field: `due_date` (raw `DueDate`). Never `close_time` / `due_time`.

```
arrival       = departure_previous + travel_time
waiting_time  = max(0, ready_time - arrival)
service_start = arrival + waiting_time      # must be <= due_date
departure     = service_start + service_time
```

Station arrival is not judged against an invented station time window. The next
`ContinueAction` will fail if the subsequent customer window is missed.

## Infeasibility

Every failed transition returns `TransitionResult.feasible = False` and an
explicit `InfeasibilityReason`:

`INSUFFICIENT_ENERGY`, `TIME_WINDOW_VIOLATION`, `CAPACITY_VIOLATION`,
`INVALID_TARGET_SOC`, `UNKNOWN_STATION`, `INVALID_STATE`, `LOOP_GUARD`,
`STATION_REVISIT`, `ZERO_CHARGE_NOOP`.

Silent success is forbidden. Global existence of a charging schedule is
**not** claimed (`charging_feasibility_status = unverified` until Part 3).

## Metrics

`TrajectoryMetrics` keeps distance, times, energy consumed / regenerated /
charged, station visits, terminal SOC, and feasibility **separate**. There is
no mixed-unit `total_cost`.

`CompletionTimeObjective` is the primary experimental objective (route
completion time). `EnergyObjective` remains the named EVRPTW-GR Model 2 energy
sum. They are never added together. There is no mixed-unit `total_cost`.

`LoadConvention` is required. Default experiments use
`OFFICIAL_REFERENCE_PICKUP`. `DELIVERY` is sensitivity only.

## Feasibility shield

See [`docs/feasibility.md`](feasibility.md). The shield masks illegal
transitions; it does not decide when to charge.

## Legacy code

The old Training/Testing stack is deleted from `main` after the new simulator
baselines exist. Git history remains the archive. Do not reimplement
price-based NS/CS/ES or mixed-unit `total_cost`.
