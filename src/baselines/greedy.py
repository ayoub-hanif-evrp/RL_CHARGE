"""Greedy charging baselines. They never use check_charging_needed."""

from __future__ import annotations

from simulation.shield import CONTINUE_INDEX, evaluate_shield, soc_interval_for_station
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy


def _detour_time(simulator: FixedRouteSimulator, station_id: str) -> float:
    current = simulator.state.current_node_id
    nxt = simulator.next_frozen_node_id()
    to_s = simulator.network.arc(current, station_id).travel_time.value
    s_to_n = simulator.network.arc(station_id, nxt).travel_time.value
    to_n = simulator.network.arc(current, nxt).travel_time.value
    return to_s + s_to_n - to_n


def _greedy_choose(simulator: FixedRouteSimulator, u: float) -> tuple[int, float]:
    from simulation.feasibility import ZERO_CHARGE_EPS
    from simulation.shield import CONTINUATION_TO_MAX, map_u_to_target_soc

    shield = evaluate_shield(simulator)
    if shield.continue_legal:
        return CONTINUE_INDEX, 0.0
    best_i = None
    best_detour = None
    for i, station_id in enumerate(shield.station_ids, start=1):
        if not shield.mask[i]:
            continue
        target = map_u_to_target_soc(simulator, station_id, u, mode=CONTINUATION_TO_MAX)
        at_station = simulator.state.current_node_id == station_id
        if at_station and target < simulator.state.soc.value + ZERO_CHARGE_EPS:
            continue
        detour = _detour_time(simulator, station_id)
        if best_detour is None or detour < best_detour:
            best_detour = detour
            best_i = i
    if best_i is None:
        return CONTINUE_INDEX, 0.0
    return best_i, u


class GreedyMinimumSufficientCharge:
    """If CONTINUE is legal, continue; else charge at the min-detour station to soc_lower."""

    name = "GreedyMinimumSufficientCharge"

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        from simulation.shield import CONTINUATION_TO_MAX

        return run_discrete_policy(
            simulator, lambda sim: _greedy_choose(sim, 0.0), soc_mode=CONTINUATION_TO_MAX
        )


class GreedyFullCharge:
    """Same station rule as min-sufficient; target soc_upper (u=1)."""

    name = "GreedyFullCharge"

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        from simulation.shield import CONTINUATION_TO_MAX

        return run_discrete_policy(
            simulator, lambda sim: _greedy_choose(sim, 1.0), soc_mode=CONTINUATION_TO_MAX
        )
