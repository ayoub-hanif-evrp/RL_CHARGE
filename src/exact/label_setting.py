"""Resource-constrained shortest path on a frozen customer sequence.

Charge decisions are the linear-charging extreme points {arrival SOC, max SOC}
plus CONTINUE. Search is exact for that action set if it finishes before the
expansion cap. Timeouts are recorded as ``timeout``, never filled with a
heuristic and never labeled exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional, Tuple

from domain.load_convention import LoadConvention
from physics.parameters import PhysicsProfile
from routing.fixed_route import FrozenRoute
from simulation.shield import action_from_discrete, evaluate_shield
from simulation.simulator import FixedRouteSimulator


@dataclass
class LabelSettingResult:
    status: str
    feasible: bool
    route_completion_time: Optional[float]
    n_expansions: int
    n_station_visits: int
    exact_for: str = "linear_charging_extreme_points"


def _state_key(sim: FixedRouteSimulator) -> Tuple:
    st = sim.state
    return (
        st.next_customer_index,
        st.current_node_id,
        st.n_station_visits_since_last_customer,
        round(st.time.value, 4),
        round(st.soc.value, 4),
    )


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[0] + 1e-9 and a[1] >= b[1] - 1e-9


def solve_label_setting(
    instance,
    route: FrozenRoute,
    *,
    profile: Optional[PhysicsProfile] = None,
    max_expansions: int = 20_000,
    load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
) -> LabelSettingResult:
    profile = profile or PhysicsProfile.from_instance(instance)
    root = FixedRouteSimulator(instance, route.customer_ids, profile, load_convention)
    heap: list = []
    counter = 0
    heappush(heap, (0.0, -root.state.soc.value, counter, root))
    best_at: dict = {}
    expansions = 0
    best_goal = None

    while heap:
        time_val, neg_soc, _, sim = heappop(heap)
        if expansions >= max_expansions:
            return LabelSettingResult(
                status="timeout",
                feasible=best_goal is not None,
                route_completion_time=None if best_goal is None else best_goal[0],
                n_expansions=expansions,
                n_station_visits=0 if best_goal is None else best_goal[1],
            )
        if sim.state.completed:
            visits = sim.state.metrics.number_of_station_visits
            if best_goal is None or time_val < best_goal[0]:
                best_goal = (time_val, visits)
            continue
        key_disc = (
            sim.state.next_customer_index,
            sim.state.current_node_id,
            sim.state.n_station_visits_since_last_customer,
        )
        resources = (sim.state.time.value, sim.state.soc.value)
        prev = best_at.get(key_disc)
        if prev is not None and _dominates(prev, resources) and prev != resources:
            continue
        if prev is None or not _dominates(resources, prev):
            best_at[key_disc] = resources
        expansions += 1
        if expansions >= max_expansions:
            return LabelSettingResult(
                status="timeout",
                feasible=best_goal is not None,
                route_completion_time=None if best_goal is None else best_goal[0],
                n_expansions=expansions,
                n_station_visits=0 if best_goal is None else best_goal[1],
            )
        shield = evaluate_shield(sim)
        if not shield.any_legal:
            continue
        candidates = []
        if shield.continue_legal:
            candidates.append((0, 0.0))
        for i, ok in enumerate(shield.mask[1:], start=1):
            if ok:
                candidates.append((i, 0.0))
                candidates.append((i, 1.0))
        for discrete, u in candidates:
            if expansions + len(heap) >= max_expansions:
                break
            nxt = sim.clone()
            result = nxt.step(action_from_discrete(nxt, discrete, u))
            if not result.feasible:
                continue
            counter += 1
            heappush(heap, (nxt.state.time.value, -nxt.state.soc.value, counter, nxt))

    if best_goal is None:
        return LabelSettingResult(
            status="infeasible",
            feasible=False,
            route_completion_time=None,
            n_expansions=expansions,
            n_station_visits=0,
        )
    return LabelSettingResult(
        status="optimal_for_action_set",
        feasible=True,
        route_completion_time=best_goal[0],
        n_expansions=expansions,
        n_station_visits=best_goal[1],
    )
