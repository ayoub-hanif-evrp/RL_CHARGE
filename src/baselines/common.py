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
    soc_mode: str = "continuation_to_max",
) -> BaselineResult:
    total = 0.0
    steps = 0
    while not simulator.state.completed:
        shield = evaluate_shield(simulator)
        t0 = simulator.state.time.value
        if not shield.any_legal:
            from simulation.feasibility import InfeasibilityReason
            from simulation.progress import failure_step_reward

            total += failure_step_reward(simulator, decision_time=t0)
            return BaselineResult(
                False,
                False,
                total,
                steps,
                extra={"dead_end": True, "reason": InfeasibilityReason.NO_FEASIBLE_ACTION},
            )
        discrete, u = choose(simulator)
        if discrete >= len(shield.mask) or not shield.mask[discrete]:
            from simulation.progress import failure_step_reward

            total += failure_step_reward(simulator, decision_time=t0)
            return BaselineResult(False, False, total, steps)
        from simulation.shield import action_from_discrete

        result = simulator.step(action_from_discrete(simulator, discrete, u, soc_mode=soc_mode))
        t1 = simulator.state.time.value
        if not result.feasible:
            from simulation.progress import failure_step_reward

            total += failure_step_reward(simulator, decision_time=t0)
            return BaselineResult(False, False, total, steps, extra={"reason": result.reason})
        total += -(t1 - t0)
        steps += 1
        if steps >= max_steps:
            return BaselineResult(False, simulator.state.completed, total, steps)
    return BaselineResult(True, True, total, steps)
