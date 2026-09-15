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
  service, then **pickup** (`payload += demand`). Advances the customer index.
- `ChargeAction(station_id, target_soc)`: travel to that station and charge
  continuously to `target_soc ∈ [current_soc, max_soc]`. Does **not** advance
  the customer index. Multiple station visits between two customers are legal.

A programming-loop guard (`loop_guard_station_visits`, default 50) exists only
to stop infinite charge cycles. It is not a scientific "maximum three stops"
constraint.

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
`INVALID_TARGET_SOC`, `UNKNOWN_STATION`, `INVALID_STATE`, `LOOP_GUARD`.

Silent success is forbidden. Global existence of a charging schedule is
**not** claimed (`charging_feasibility_status = unverified` until Part 3).

## Metrics

`TrajectoryMetrics` keeps distance, times, energy consumed / regenerated /
charged, station visits, terminal SOC, and feasibility **separate**. There is
no mixed-unit `total_cost`.

`EnergyObjective` is the named EVRPTW-GR-compatible objective: the sum of net
arc energies (Model 2 `sum(hh * distance)`).

## Legacy code

The following Training/Testing modules still exist on disk for Git history and
will be replaced in Part 3. Do not import them from the new scientific core:

- `Training/environment.py`, `helper_functions.py`, `matrices_creator.py` and
  the identical Testing copies (1:1:1 distance=time=energy, rule-based charge
  trigger, six-level portions)
- `Training/ddqn_agent.py`, `neural_networks.py`
- `Testing/baslines_comp/*`
- `fleet.json` (synthetic)
- `Dataset_splitting.ipynb`

Official Model 2 allows at most one station between two customers. This
simulator does not inherit that MILP restriction.
