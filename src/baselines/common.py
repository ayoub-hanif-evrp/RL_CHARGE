"""Shared baseline rollout helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from simulation.shield import CONTINUE_INDEX, evaluate_shield
from simulation.simulator import FixedRouteSimulator


@dataclass
class BaselineResult:
    feasible: bool
    completed: bool
    return_value: float
    n_steps: int
    extra: dict = field(default_factory=dict)


def run_discrete_policy(
    simulator: FixedRouteSimulator,
    choose: Callable[[FixedRouteSimulator], tuple[int, float]],
    *,
    max_steps: int = 10_000,
    soc_mode: str = "arrival_to_max",
) -> BaselineResult:
    total = 0.0
    steps = 0
    while not simulator.state.completed:
        shield = evaluate_shield(simulator)
        if not shield.any_legal:
            total += -(simulator.horizon - simulator.state.time.value)
            return BaselineResult(False, False, total, steps, extra={"dead_end": True})
        t0 = simulator.state.time.value
        discrete, u = choose(simulator)
        if discrete >= len(shield.mask) or not shield.mask[discrete]:
            total += -(simulator.horizon - t0)
            return BaselineResult(False, False, total, steps)
        from simulation.shield import action_from_discrete

        result = simulator.step(action_from_discrete(simulator, discrete, u, soc_mode=soc_mode))
        t1 = simulator.state.time.value
        if not result.feasible:
            total += -(simulator.horizon - t0)
            return BaselineResult(False, False, total, steps, extra={"reason": result.reason})
        total += -(t1 - t0)
        steps += 1
        if steps >= max_steps:
            return BaselineResult(False, simulator.state.completed, total, steps)
    return BaselineResult(True, True, total, steps)
