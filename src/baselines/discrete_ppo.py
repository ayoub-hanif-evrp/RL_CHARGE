"""Discrete-PPO baseline: categorical station + six charge levels. No Beta snap.

The simulator remains continuous. Charge-level probabilities are conditioned
on the selected station. CONTINUE uses only the station/CONTINUE categorical.
"""

from __future__ import annotations

from typing import Sequence

import torch

from rl.ablation import AblationConfig
from rl.policy import CHARGE_U_LEVELS, HybridPolicy
from simulation.simulator import FixedRouteSimulator

from .common import BaselineResult, run_discrete_policy

DISCRETE_U_GRID = CHARGE_U_LEVELS


def snap_u(u: float, grid: Sequence[float] = DISCRETE_U_GRID) -> float:
    """Nearest grid point. Not used by DiscretePPO sampling; kept for tests."""
    return float(min(grid, key=lambda value: abs(value - u)))


class DiscretePPO:
    name = "DiscretePPO"

    def __init__(self, policy: HybridPolicy, grid: Sequence[float] = DISCRETE_U_GRID):
        if not policy.ablation.discrete_u:
            policy.ablation = AblationConfig.from_name("A1")
        self.policy = policy
        self.grid = tuple(grid)

    def choose(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> tuple[int, float]:
        from rl.features import extract_features

        soc = getattr(self.policy.ablation, "soc_interval", "continuation_to_max")
        features = extract_features(simulator, soc_interval=soc)
        with torch.no_grad():
            output = self.policy.act(features, eval_mode=eval_mode)
        discrete = int(output.discrete_index.item())
        u = float(output.u.item())
        if discrete == 0:
            u = 0.0
        return discrete, u

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(simulator, lambda sim: self.choose(sim, eval_mode=eval_mode))
