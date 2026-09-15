"""Discrete-PPO baseline: same encoder, station SOC snapped to a documented grid.

The grid lives only in this agent. The physics engine still accepts continuous SOC.
"""

from __future__ import annotations

from typing import Sequence

import torch

from rl.policy import HybridPolicy
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy

# Documented agent-side grid. Not a physics constraint.
DISCRETE_U_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def snap_u(u: float, grid: Sequence[float] = DISCRETE_U_GRID) -> float:
    return float(min(grid, key=lambda value: abs(value - u)))


class DiscretePPO:
    name = "DiscretePPO"

    def __init__(self, policy: HybridPolicy, grid: Sequence[float] = DISCRETE_U_GRID):
        self.policy = policy
        self.grid = tuple(grid)

    def choose(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> tuple[int, float]:
        from rl.features import extract_features

        features = extract_features(simulator)
        with torch.no_grad():
            output = self.policy.act(features, eval_mode=eval_mode)
        discrete = int(output.discrete_index.item())
        u = snap_u(float(output.u.item()), self.grid)
        if discrete == 0:
            u = 0.0
        return discrete, u

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(simulator, lambda sim: self.choose(sim, eval_mode=eval_mode))
