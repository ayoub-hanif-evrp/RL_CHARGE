"""Restricted label-setting search on a frozen customer sequence.

Charge decisions are CONTINUE plus the linear-charging extreme points
``{arrival SOC, max SOC}``. This is **not** exact for continuous Hybrid PPO.
Timeouts are recorded as ``timeout`` and never filled with a heuristic.

A feasible finish is a lower bound on charging feasibility for this restricted
action set. Infeasible/timeout does **not** prove the continuous problem is
infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter
from typing import List, Optional, Tuple

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
    exact_for: str = "restricted_linear_charging_extreme_points"


def _discrete_key(sim: FixedRouteSimulator) -> Tuple:
    st = sim.state
    visited = tuple(sorted(st.stations_visited_since_progress))
    return (st.next_customer_index, st.current_node_id, visited)


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """``a`` weakly dominates ``b``: no worse time and no worse SOC."""
    return a[0] <= b[0] + 1e-9 and a[1] >= b[1] - 1e-9


def _pareto_accept(frontier: List[Tuple[float, float]], resources: Tuple[float, float]) -> bool:
    """Keep a Pareto set of (time, SOC). Lower time and higher SOC are better."""
    for prev in frontier:
        if _dominates(prev, resources):
            return False
    kept = [prev for prev in frontier if not _dominates(resources, prev)]
    kept.append(resources)
    frontier.clear()
    frontier.extend(kept)
    return True


def solve_label_setting(
    instance,
    route: FrozenRoute,
    *,
    profile: Optional[PhysicsProfile] = None,
    max_expansions: int = 20_000,
    load_convention=LoadConvention.OFFICIAL_REFERENCE_PICKUP,
    stop_at_first_feasible: bool = False,
    max_seconds: Optional[float] = None,
) -> LabelSettingResult:
    profile = profile or PhysicsProfile.from_instance(instance)
    root = FixedRouteSimulator(instance, route.customer_ids, profile, load_convention)
    heap: list = []
    counter = 0
    heappush(heap, (0.0, -root.state.soc.value, counter, root))
    best_at: dict[Tuple, List[Tuple[float, float]]] = {}
    expansions = 0
    best_goal = None
    t_start = perf_counter()

    def _timeout() -> LabelSettingResult:
        return LabelSettingResult(
            status="timeout",
            feasible=best_goal is not None,
            route_completion_time=None if best_goal is None else best_goal[0],
            n_expansions=expansions,
            n_station_visits=0 if best_goal is None else best_goal[1],
        )

    while heap:
        if max_seconds is not None and (perf_counter() - t_start) >= float(max_seconds):
            return _timeout()
        time_val, neg_soc, _, sim = heappop(heap)
        if expansions >= max_expansions:
            return _timeout()
        if sim.state.completed:
            visits = sim.state.metrics.number_of_station_visits
            if best_goal is None or time_val < best_goal[0]:
                best_goal = (time_val, visits)
            if stop_at_first_feasible:
                return LabelSettingResult(
                    status="feasible_for_action_set",
                    feasible=True,
                    route_completion_time=best_goal[0],
                    n_expansions=expansions,
                    n_station_visits=best_goal[1],
                )
            continue
        resources = (sim.state.time.value, sim.state.soc.value)
        frontier = best_at.setdefault(_discrete_key(sim), [])
        if not _pareto_accept(frontier, resources):
            continue
        expansions += 1
        if expansions >= max_expansions:
            return _timeout()
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
