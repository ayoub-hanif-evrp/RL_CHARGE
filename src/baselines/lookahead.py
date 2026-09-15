"""One-step lookahead over CONTINUE and unmasked station endpoint SOCs."""

from __future__ import annotations

from simulation.shield import (
    CONTINUE_INDEX,
    action_from_discrete,
    evaluate_shield,
)
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy


def _lookahead_choose(simulator: FixedRouteSimulator) -> tuple[int, float]:
    shield = evaluate_shield(simulator)
    best = None
    if shield.continue_legal:
        probe = simulator.clone()
        result = probe.step(action_from_discrete(probe, CONTINUE_INDEX, 0.0))
        if result.feasible:
            best = (probe.state.time.value, CONTINUE_INDEX, 0.0)
    for i, _station_id in enumerate(shield.station_ids, start=1):
        if not shield.mask[i]:
            continue
        for u in (0.0, 1.0):
            probe = simulator.clone()
            result = probe.step(action_from_discrete(probe, i, u))
            if not result.feasible:
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
        return run_discrete_policy(simulator, _lookahead_choose)
