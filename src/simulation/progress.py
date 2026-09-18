"""Method-independent frozen-route remaining-time lower bound.

This is a horizon-derived failure-progress term in TIME units. It is not a
second operational objective. Charging time and customer waiting are omitted
so the quantity remains a lower bound on remaining completion time.
"""

from __future__ import annotations

from simulation.simulator import FixedRouteSimulator


def remaining_time_lower_bound(simulator: FixedRouteSimulator) -> float:
    """Conservative lower bound on remaining frozen-route completion time.

    Includes:
    - direct travel from the current node to the next frozen customer (or depot)
    - direct travel along the remaining frozen customer order
    - remaining customer service times
    - final direct return to the depot

    Excludes charging duration and waiting. Uses only the current simulator
    geometry and the immutable customer sequence. Never reads VAL/TEST files.
    """
    if simulator.state.completed:
        return 0.0
    current = simulator.state.current_node_id
    remaining = list(simulator.remaining_customer_ids())
    depot = simulator.depot_id
    total = 0.0
    if remaining:
        nxt = remaining[0]
        total += simulator.network.arc(current, nxt).travel_time.value
        total += float(simulator.network.node(nxt).service_time)
        prev = nxt
        for cid in remaining[1:]:
            total += simulator.network.arc(prev, cid).travel_time.value
            total += float(simulator.network.node(cid).service_time)
            prev = cid
        total += simulator.network.arc(prev, depot).travel_time.value
    elif current != depot:
        total += simulator.network.arc(current, depot).travel_time.value
    return max(0.0, float(total))


def failure_step_reward(simulator: FixedRouteSimulator) -> float:
    """Terminal failure reward: ``-(H - t) - L_remaining(state)``."""
    t = float(simulator.state.time.value)
    h = float(simulator.horizon)
    return -(h - t) - remaining_time_lower_bound(simulator)
