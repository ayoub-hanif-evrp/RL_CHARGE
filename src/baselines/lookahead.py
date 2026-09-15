"""One-step lookahead over CONTINUE and meaningful unmasked station endpoints."""

from __future__ import annotations

from simulation.feasibility import ZERO_CHARGE_EPS
from simulation.shield import (
    CONTINUE_INDEX,
    CONTINUATION_TO_MAX,
    action_from_discrete,
    evaluate_shield,
    map_u_to_target_soc,
)
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy


def _distance_sq_to_next(simulator: FixedRouteSimulator, node_id: str) -> float:
    nxt = simulator.network.node(simulator.next_frozen_node_id())
    node = simulator.network.node(node_id)
    return (node.x - nxt.x) ** 2 + (node.y - nxt.y) ** 2


def _progresses_toward_next(simulator: FixedRouteSimulator, station_id: str) -> bool:
    here = _distance_sq_to_next(simulator, simulator.state.current_node_id)
    dest = _distance_sq_to_next(simulator, station_id)
    return dest < here - 1e-12


def _meaningful_station_action(simulator: FixedRouteSimulator, station_id: str, u: float) -> bool:
    """Skip zero-energy / no-progress station actions (same-station no-ops)."""
    target = map_u_to_target_soc(simulator, station_id, u, mode=CONTINUATION_TO_MAX)
    at_station = simulator.state.current_node_id == station_id
    if at_station and target < simulator.state.soc.value + ZERO_CHARGE_EPS:
        return False
    return True


def _lookahead_choose(simulator: FixedRouteSimulator) -> tuple[int, float]:
    shield = evaluate_shield(simulator)
    best = None
    if shield.continue_legal:
        probe = simulator.clone()
        result = probe.step(action_from_discrete(probe, CONTINUE_INDEX, 0.0, soc_mode=CONTINUATION_TO_MAX))
        if result.feasible:
            best = (probe.state.time.value, CONTINUE_INDEX, 0.0)
    for i, station_id in enumerate(shield.station_ids, start=1):
        if not shield.mask[i]:
            continue
        for u in (0.0, 1.0):
            if not _meaningful_station_action(simulator, station_id, u):
                continue
            if shield.continue_legal and not _progresses_toward_next(simulator, station_id):
                continue
            probe = simulator.clone()
            result = probe.step(action_from_discrete(probe, i, u, soc_mode=CONTINUATION_TO_MAX))
            if not result.feasible:
                continue
            charged = result.state.metrics.total_energy_charged
            prior = simulator.state.metrics.total_energy_charged
            at_station = simulator.state.current_node_id == station_id
            if at_station and charged < prior + ZERO_CHARGE_EPS:
                continue
            visits = result.state.metrics.number_of_station_visits
            if visits > simulator.state.metrics.number_of_station_visits and charged < prior + ZERO_CHARGE_EPS:
                continue
            score = probe.state.time.value
            if best is None or score < best[0]:
                best = (score, i, u)
    if best is None:
        legal = next((i for i, ok in enumerate(shield.mask) if ok), CONTINUE_INDEX)
        return legal, 0.0
    return best[1], best[2]


class OneStepLookahead:
    name = "OneStepLookahead"

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(
            simulator, _lookahead_choose, soc_mode=CONTINUATION_TO_MAX
        )
