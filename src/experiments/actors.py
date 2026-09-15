"""Learned-policy adapters for batch evaluation."""

from __future__ import annotations

import torch

from baselines.common import BaselineResult, run_discrete_policy
from baselines.discrete_ppo import snap_u
from rl.ablation import AblationConfig
from rl.features import extract_features
from rl.normalization import Normalizer
from rl.policy import HybridPolicy
from simulation.simulator import FixedRouteSimulator


class HybridPolicyActor:
    name = "HybridPPO"

    def __init__(
        self,
        policy: HybridPolicy,
        ablation: AblationConfig | None = None,
        snap_discrete_u: bool = False,
        eval_mode: bool = True,
        normalizer: Normalizer | None = None,
    ):
        self.policy = policy
        self.policy.eval()
        self.ablation = ablation or AblationConfig()
        self.snap_discrete_u = snap_discrete_u
        self.eval_mode = eval_mode
        self.normalizer = normalizer

    def choose(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> tuple[int, float]:
        features = extract_features(
            simulator,
            self.normalizer,
            use_remaining_route=self.ablation.use_remaining_route,
            use_terrain_load_features=self.ablation.use_terrain_load_features,
            soc_interval=self.ablation.soc_interval,
        )
        with torch.no_grad():
            output = self.policy.act(features, eval_mode=eval_mode)
        discrete = int(output.discrete_index.item())
        u = float(output.u.item())
        if self.snap_discrete_u or self.ablation.discrete_u:
            u = snap_u(u)
        if discrete == 0:
            u = 0.0
        return discrete, u

    def __call__(self, simulator: FixedRouteSimulator, eval_mode: bool = True) -> BaselineResult:
        return run_discrete_policy(
            simulator,
            lambda sim: self.choose(sim, eval_mode=eval_mode),
            soc_mode=self.ablation.soc_interval,
        )
